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
    """The same vector through SGLang, one request at a time and then together."""
    params = {"max_new_tokens": 1, "temperature": 0}
    if batched:
        outputs = engine.generate(prompts, sampling_params=params, return_hidden_states=True)
    else:
        outputs = [engine.generate(p, sampling_params=params, return_hidden_states=True)
                   for p in prompts]
    out = []
    for item in outputs:
        states = np.asarray(item["meta_info"]["hidden_states"], dtype=np.float32)
        # This engine returns [generated_step, position, width]: one layer, every
        # position. The readout is the last prompt position. A build that returns a
        # depth dimension would need a different index, so the rank is checked
        # rather than assumed.
        if states.ndim == 3 and states.shape[0] == 1:
            out.append(states[0, -1])
        elif states.ndim == 2:
            out.append(states[-1])
        else:
            raise SystemExit(f"unexpected hidden_states shape {states.shape}; "
                             "this build needs a different readout index")
    return np.stack(out), states.ndim


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
    """Pass one: the reference vectors, where transformers knows this architecture."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    reference = hf_hidden(model, tokenizer, PROMPTS, args.layer)
    np.savez_compressed(args.reference_out, reference=reference, layer=args.layer,
                        model=args.model, prompts=np.array(PROMPTS, dtype=object),
                        transformers=__import__("transformers").__version__)
    print(f"wrote {reference.shape} from layer {args.layer} to {args.reference_out}")


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
