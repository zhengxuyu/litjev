"""Check that SGLang hands back the hidden state the head was trained on.

Run this against any new engine version, backbone or layer before serving a
head through `inference`. It is not specific to any benchmark: the
question is only whether one vector arrives intact.

The head reads one vector: layer L at the last prompt position. If SGLang
returns a different layer, a different position, or the right vector attached to
the wrong request in a batch, nothing fails. The head simply reads noise, the
rollout completes, and the numbers mean nothing. SGLang has open issues for
exactly this under batching (sgl-project/sglang#8066, #4997), so this runs
before anything is built on top.

The reference is transformers, which is what produced every recorded hidden
state so far. Agreement is checked at batch size one and again batched, because
the reported bug only appears when several requests share a forward pass.

It also checks the content readout, which is the second thing SGLang can return
differently without failing. That readout sums the log-probabilities the model
gave an option's own tokens, so it depends on `logprob_start_len` meaning what
this code thinks it means and on the engine scoring the ids it was handed. A
wrong offset there shifts every option by one position, which leaves the ranking
plausible and the numbers wrong. The sums themselves will not match to the last
decimal -- the two implementations differ by a few percent on a single
activation, and a sum over several positions inherits that -- so what is required
is that the two choose the same option, which is the only part a decision reads.

It runs in two passes, because the two sides cannot share an environment:
sglang pins transformers to a version that does not recognise this backbone's
architecture, while the reference forward pass needs one that does. So
`--reference-out` computes the vectors where transformers knows the model and
writes them to a file, and `--reference-in` compares SGLang against that file
where SGLang lives. Neither pin has to be broken.
"""
import argparse
import json

import numpy as np
import torch

PROMPTS = [
    "Question: which is larger?\nA two\nB three\nAnswer:",
    (
        "Question: a longer prompt, because a batching bug that pads or reorders shows up "
        "between a short request and a long one rather than between two of equal length. "
        "Consider a sequence of tokens long enough to occupy several blocks of the engine's "
        "paged attention, and then ask something simple about it.\nA yes\nB no\nC maybe\n"
        "D none of these\nAnswer:"
    ),
    "Short.\nAnswer:",
    (
        "Question: and one more of middling length, so three distinct lengths are present and "
        "the batch is not symmetric.\nA alpha\nB beta\nAnswer:"
    ),
]


def option_questions():
    """What the content readout is checked on, built where the schema is importable.

    Option lengths are deliberately uneven within a question. An off-by-one in
    the offset costs every option one token's log-probability, so options of equal
    length would hide it in a constant that cancels in the ranking; uneven ones do
    not, and the longest option is where it is worst.
    """
    from litjev.schema import Choice

    return [
        ("A customer writes: my order arrived broken and I need it replaced today.",
         Choice(instructions="What does the customer want?",
                criteria={"refund": "money back", "replace": "a replacement",
                          "info": "just information about the order and nothing else"})),
        ("Oxygen is element 8.",
         Choice(instructions="Which element is this?",
                criteria={"a": "oxygen", "b": "osmium", "c": "a gas that makes up about "
                          "twenty-one percent of the atmosphere by volume"})),
        ("",
         Choice(instructions="Which is larger?",
                criteria={"a": "two", "b": "three"})),
    ]


def option_scores(scorer):
    """`(index, tokens, sum)` per option, per question, through either backend."""
    table = []
    for state, question in option_questions():
        scores, _ = scorer.score_options(state, question, content_free=False)
        table.append([(score.index, score.tokens, float(score.total)) for score in scores])
    return table


def report_options(reference, candidate):
    """Whether the two backends read the same answer off the same options.

    The sums are reported and not thresholded, for the reason `head_disagreement`
    gives about the vectors: a tolerance on a float is a number somebody picks.
    What is checked is the token counts, which must be identical because the same
    tokenizer produced both and a difference means the prompts differ, and the
    chosen option, which is what the readout is for.
    """
    ok = True
    worst_sum = 0.0
    per_question = []
    for index, (want, got) in enumerate(zip(reference, candidate, strict=True)):
        if [row[1] for row in want] != [row[1] for row in got]:
            print(f"{'option tokens':26} question {index}: {[row[1] for row in want]} "
                  f"against {[row[1] for row in got]}; the prompts differ")
            ok = False
            continue
        sums_reference = [row[2] for row in want]
        sums_candidate = [row[2] for row in got]
        worst_sum = max(worst_sum, max(abs(a - b) for a, b
                                       in zip(sums_reference, sums_candidate)))
        # The margin beside the choice, because a flip on two options a hair apart
        # is a tie resolved differently and a flip on a wide margin is a fault, and
        # the line that reports only the choice cannot be told which one happened.
        margins = [_top_two_margin(sums_reference), _top_two_margin(sums_candidate)]
        print(f"{'':26} question {index}: chose {int(np.argmax(sums_reference))} and "
              f"{int(np.argmax(sums_candidate))}, margin {margins[0]:.4f} and "
              f"{margins[1]:.4f}")
        if int(np.argmax(sums_reference)) != int(np.argmax(sums_candidate)):
            print(f"{'content readout':26} question {index}: transformers chooses "
                  f"{int(np.argmax(sums_reference))}, SGLang chooses "
                  f"{int(np.argmax(sums_candidate))}, with margins "
                  f"{margins[0]:.4f} and {margins[1]:.4f} -- a margin near zero is a "
                  f"tie resolved differently, not a misalignment")
            ok = False
        per_question.append({"question": index, "margin_reference": margins[0],
                             "margin_candidate": margins[1],
                             "chosen_reference": int(np.argmax(sums_reference)),
                             "chosen_candidate": int(np.argmax(sums_candidate))})
    print(f"{'content readout':26} |sum difference| max {worst_sum:.5f}   "
          f"{'same choice everywhere' if ok else 'THE CHOICE MOVED'}")
    return ok, {"sum_difference_max": float(worst_sum), "same_choice": bool(ok),
                "questions": len(reference), "per_question": per_question}


def _top_two_margin(scores):
    """How much the chosen option won by. Zero when there is nothing to choose."""
    if len(scores) < 2:
        return 0.0
    best, second = sorted(scores, reverse=True)[:2]
    return float(best - second)


def hf_hidden(model, tokenizer, prompts, layer):
    """Layer `layer` at the last prompt position, one prompt at a time."""
    out = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            result = model(**ids, output_hidden_states=True)
        out.append(result.hidden_states[layer][0, -1].float().cpu().numpy())
    return np.stack(out)


def sglang_hidden(engine, prompts, layer, batched):
    """The same vector through SGLang, one request at a time and then together.

    Refuses any layer but the last rather than accepting one it cannot honour.
    This engine returns `[step, position, width]` with no depth axis, so there is
    nothing for `--layer 48` to select; it used to be accepted and ignored, and
    the comparison then reported MISALIGNED for a reason that was not real --
    transformers taking `hidden_states[48]` against SGLang's last layer.
    """
    from inference.sglang_backend import readout_hidden_state

    if layer != -1:
        raise SystemExit(
            f"--layer {layer}: this engine returns one layer and no depth axis, so only "
            f"-1 can be compared. A head on another layer cannot be served through "
            f"SGLang either, which is the finding, not a limitation of this script")
    params = {"max_new_tokens": 1, "temperature": 0}
    if batched:
        outputs = engine.generate(prompts, sampling_params=params, return_hidden_states=True)
    else:
        outputs = [engine.generate(p, sampling_params=params, return_hidden_states=True)
                   for p in prompts]
    out, ndim = [], 0
    for item in outputs:
        states = item["meta_info"]["hidden_states"]
        ndim = np.asarray(states, dtype=np.float32).ndim
        out.append(readout_hidden_state(states))
    return np.stack(out), ndim


def head_disagreement(head, reference, candidate, lambdas=(0.1, 0.05, 0.0, -0.05, -0.1)):
    """How far apart the head's decision is on the two vectors.

    The vectors themselves differ by a few percent between implementations, which
    is what different fused kernels and accumulation orders do to a bf16 activation
    after sixty-four layers. A tolerance on the vector is therefore a threshold
    somebody has to pick. What matters instead is measurable: the gain the head
    reads off it, and whether that gain lands on the same side of lambda.
    """
    from litjev.heads import build_features
    from litjev.routing import escalation_gain, fast_confidence

    # A flat candidate distribution, so the statistics contribute the same thing to
    # both sides and only the hidden state differs, which is what is being compared.
    flat = np.full(4, 0.25)
    gains, confidences = {}, {}
    for name, vectors in (("reference", reference), ("sglang", candidate)):
        outcome = head.predict(np.stack([build_features(v, flat, "choice") for v in vectors]))
        gains[name] = np.array([escalation_gain(row) for row in outcome])
        confidences[name] = np.array([fast_confidence(row) for row in outcome])
    delta = np.abs(gains["reference"] - gains["sglang"])
    flips = {
        str(lam): int(np.sum((gains["reference"] > lam) != (gains["sglang"] > lam)))
        for lam in lambdas
    }
    print(f"{'head decision':26} |gain difference| max {delta.max():.5f} mean {delta.mean():.5f}"
          f"   confidence max {np.abs(confidences['reference'] - confidences['sglang']).max():.5f}")
    print(f"{'':26} escalation flips per lambda: {flips}")
    return {"gain_difference_max": float(delta.max()),
            "gain_difference_mean": float(delta.mean()),
            "confidence_difference_max": float(
                np.abs(confidences["reference"] - confidences["sglang"]).max()),
            "escalation_flips": flips,
            "gains_reference": gains["reference"].tolist(),
            "gains_sglang": gains["sglang"].tolist()}


def report(name, reference, candidate):
    cosine = np.sum(reference * candidate, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1) + 1e-12)
    error = np.linalg.norm(reference - candidate, axis=1) / (
        np.linalg.norm(reference, axis=1) + 1e-12)
    # Direction has to match. Magnitude is left to head_disagreement, which asks
    # whether the difference reaches the decision instead of comparing it to a
    # number somebody chose.
    ok = bool(cosine.min() > 0.999)
    print(f"{name:26} cosine min {cosine.min():.5f}  relative error max {error.max():.5f}  "
          f"{'aligned' if ok else 'MISALIGNED'}")
    # A batching bug usually shows up as one request holding another's vector, so
    # also check that no candidate matches the wrong reference better than its own.
    gram = reference @ candidate.T
    gram /= np.linalg.norm(reference, axis=1)[:, None] * np.linalg.norm(candidate, axis=1)[None, :]
    mismatched = [int(i) for i in range(len(gram)) if int(np.argmax(gram[i])) != i]
    if mismatched:
        print(f"{'':26} request(s) {mismatched} match another prompt's vector better than their own")
        ok = False
    return ok


def write_reference(args):
    """Pass one: the reference vectors, where transformers knows this architecture.

    Loaded through `TransformersScorer`, not through `AutoModelForCausalLM`
    directly, because the loader is not the same for every backbone this compares:
    a native vision-language checkpoint needs `AutoModelForImageTextToText` and
    refuses the causal-LM loader. The scorer picks it from the config, and it is
    the same choice that produced every recorded number, so the reference is from
    the same object the experiments used rather than from one assembled here.
    """
    from litjev.backend import ModelSettings, TransformersScorer

    scorer = TransformersScorer.load(ModelSettings(
        args.model, revision="main", device_map="cuda:0", dtype="bfloat16"))
    model, tokenizer = scorer.model.eval(), scorer.tokenizer
    reference = hf_hidden(model, tokenizer, PROMPTS, args.layer)
    options = "[]"
    if not args.skip_options:
        options = json.dumps(option_scores(scorer))
    np.savez_compressed(args.reference_out, reference=reference, layer=args.layer,
                        model=args.model, prompts=np.array(PROMPTS, dtype=object),
                        options=options,
                        transformers=__import__("transformers").__version__)
    print(f"wrote {reference.shape} from layer {args.layer} to {args.reference_out}"
          + ("" if args.skip_options else f", and {len(json.loads(options))} option questions"))


def compare(args):
    """Pass two: SGLang against the file, where SGLang lives."""
    stored = np.load(args.reference_in, allow_pickle=True)
    reference = stored["reference"]
    if int(stored["layer"]) != args.layer:
        raise SystemExit(f"the reference is layer {int(stored['layer'])}, not {args.layer}")
    if str(stored["model"]) != args.model:
        raise SystemExit(f"the reference is for {stored['model']}, not {args.model}")
    print(f"reference {reference.shape} from layer {args.layer}, "
          f"transformers {stored['transformers']}")

    import sglang as sgl

    engine = sgl.Engine(model_path=args.model, enable_return_hidden_states=True,
                        mem_fraction_static=0.75)
    single, ndim = sglang_hidden(engine, PROMPTS, args.layer, batched=False)
    results = {"one_at_a_time": report("one request at a time", reference, single)}
    batched, _ = sglang_hidden(engine, PROMPTS, args.layer, batched=True)
    results["batched"] = report("four in one batch", reference, batched)
    results.update({"model": args.model, "layer": args.layer, "prompts": len(PROMPTS),
                    "hidden_states_ndim": int(ndim)})

    options_agree = True
    stored_options = json.loads(str(stored["options"])) if "options" in stored else []
    if args.skip_options:
        print("content readout: skipped")
    elif not stored_options:
        raise SystemExit("the reference holds no option scores; write it again without "
                         "--skip-options, or pass --skip-options here too")
    else:
        from transformers import AutoTokenizer

        from inference.sglang_backend import SGLangScorer, SGLangSettings

        scorer = SGLangScorer(
            engine, AutoTokenizer.from_pretrained(args.model),
            SGLangSettings(model_id=args.model, feature_layers=()), hidden_size=0)
        options_agree, results["options"] = report_options(stored_options,
                                                           option_scores(scorer))

    if args.decision_head:
        from litjev.heads import DecisionHead

        head = DecisionHead.load(args.decision_head)
        if tuple(head.metadata.feature_layers) != (args.layer,):
            raise SystemExit(f"the head reads {head.metadata.feature_layers}, not layer {args.layer}")
        results["decision"] = head_disagreement(head, reference, single)
        results["decision_batched"] = head_disagreement(head, reference, batched)
        flips = sum(int(v) for v in results["decision"]["escalation_flips"].values())
        flips += sum(int(v) for v in results["decision_batched"]["escalation_flips"].values())
        results["escalation_flips_total"] = flips
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)

    if not (results["one_at_a_time"] and results["batched"]):
        raise SystemExit("SGLang's hidden states point elsewhere than transformers'; "
                         "do not build on this")
    if not options_agree:
        raise SystemExit("the content readout reads a different answer on SGLang than on "
                         "transformers; do not serve it or put its numbers in a table")
    if args.decision_head and results["escalation_flips_total"] > 0:
        raise SystemExit(f"{results['escalation_flips_total']} escalation decisions differ "
                         "between the two implementations; the gap reaches the routing")
    print("\nthe two implementations agree where it matters")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--reference-out", help="pass one: write the transformers reference here")
    parser.add_argument("--reference-in", help="pass two: compare SGLang against this file")
    parser.add_argument("--output", help="pass two: where the comparison goes")
    parser.add_argument("--decision-head",
                        help="pass two: also compare what this head decides on both vectors")
    parser.add_argument("--skip-options", action="store_true",
                        help="leave the content readout unchecked; both passes must agree "
                             "on this, since pass two needs what pass one wrote")
    args = parser.parse_args()
    if bool(args.reference_out) == bool(args.reference_in):
        parser.error("give exactly one of --reference-out or --reference-in")
    if args.reference_out:
        write_reference(args)
    else:
        if not args.output:
            parser.error("--reference-in needs --output")
        compare(args)


if __name__ == "__main__":
    main()
