"""The hybrid-thinking model's two modes, served by SGLang.

Implements the same surface as `litjev.backend.TransformersScorer`:

    hidden_size                              the backbone's width
    score(state, schema)                     the fast mode under the coded readout
    score_options(state, question)           the fast mode under the content readout
    think(state, schema, names, budget)      the slow mode, for the named questions

so `SchemaDecisionEngine` takes either without knowing which.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from litjev.content_readout import OptionScore, option_token_ids
from litjev.decision import RawFieldScores
from litjev.prompting import (
    ANSWER_BOUNDARY,
    CONTENT_FREE_INSTRUCTIONS,
    build_content_messages,
    build_decision_messages,
    build_thinking_messages,
    listable_options,
    question_suffix,
    unlabelled_suffix,
)
from litjev.slots import SLOT_FORMAT, candidate_codes


@dataclass(frozen=True)
class SGLangSettings:
    """What the engine needs, and the two choices that are ours rather than its.

    `feature_layers` is the layer the decision head reads. An empty tuple turns
    hidden-state extraction off, which is cheaper and is the right setting when
    no head is being served.

    `batch_questions` sends a request's questions to the engine together, which
    is what the prefix cache rewards. SGLang has open issues where batched
    hidden states are attributed to the wrong request (sgl-project/sglang#8066,
    #4997), and that failure is silent: the head reads another question's state
    and the run completes. Set it false to serve questions one at a time, and
    run `inference.verify` on any engine version before trusting true.
    """

    model_id: str
    revision: str = "main"
    feature_layers: tuple[int, ...] = ()
    batch_questions: bool = False
    mem_fraction_static: float = 0.8
    max_input_tokens: int = 16384
    engine_kwargs: dict = field(default_factory=dict)


def readout_hidden_state(states, feature_layers=(-1,)):
    """The vector at the last prompt position, from what this engine returns.

    One parser, because there were two and they had diverged. This module read a
    three-dimensional return as `[position, layer, width]` and `inference.verify`
    read it as `[step, position, width]`. Measuring settled it rather than the
    rank: a 15-token prompt gives `[1, 15, 5120]`, so the axis that moves
    with the prompt is the middle one and there is **no depth axis at all**.

    Under the wrong reading `feature_layers=(-1,)` was right by coincidence --
    the last of a position axis read as layers is the last position -- while
    `(48,)` silently returned the vector at position 48, or raised IndexError on
    a prompt shorter than 49 tokens. A head on any layer but the last therefore
    cannot be served here, and that is refused rather than approximated.
    """
    states = np.asarray(states, dtype=np.float32)
    if tuple(feature_layers) not in ((), (-1,)):
        raise RuntimeError(
            f"this engine returns one layer and no depth axis, so the head's "
            f"{list(feature_layers)} cannot be read from it: serve that head on "
            f"transformers, or train one on layer -1")
    if states.ndim == 3:                     # [step, position, width], as measured
        return states[0, -1]
    if states.ndim == 2:                     # [position, width]; not seen on this build
        return states[-1]
    raise RuntimeError(f"unexpected hidden_states shape {states.shape}; "
                       f"vary the prompt length to see which axis moves with it")


class SGLangScorer:
    """Fast and slow modes over one SGLang engine."""

    def __init__(self, engine, tokenizer, settings, hidden_size):
        # Also here, for every way a scorer is built that is not `load`: the engine
        # is passed in by `SGLangServerScorer`, by `verify`, and by the tests.
        self._check_layers(settings)
        self.engine = engine
        self.tokenizer = tokenizer
        self.settings = settings
        self._hidden_size = hidden_size

    @staticmethod
    def _hidden_size(model_id, revision):
        """The backbone's width, read from the checkpoint rather than asked of transformers.

        SGLang pins a transformers that does not recognise every architecture it can
        itself serve, so `AutoConfig` refuses checkpoints the engine loads happily.
        The width is a number in config.json; reading it needs nobody's opinion of
        the architecture.
        """
        import json
        from pathlib import Path

        path = Path(model_id) / "config.json"
        if path.exists():
            config = json.loads(path.read_text())
        else:
            from huggingface_hub import hf_hub_download

            config = json.loads(Path(hf_hub_download(
                model_id, "config.json", revision=revision)).read_text())
        for node in (config.get("text_config", {}), config):
            if "hidden_size" in node:
                return int(node["hidden_size"])
        raise ValueError(f"no hidden_size in the config of {model_id}")

    @staticmethod
    def _check_layers(settings):
        """Refuse a head this engine cannot serve, before anything has been loaded.

        It was in `__init__` alone, which `load` reaches only after
        `sgl.Engine(...)` has loaded the backbone -- so on the real path the
        refusal arrived after the 27B load it exists to save.
        """
        if tuple(settings.feature_layers) not in ((), (-1,)):
            raise ValueError(
                f"this engine returns only the last hidden layer, so a head reading "
                f"{list(settings.feature_layers)} cannot be served through it. Train one "
                f"with litjev-train-head --candidate layer_-1, or serve it on transformers")

    @classmethod
    def load(cls, settings):
        # First, before `import sglang` and before the config is read, so that a
        # head this engine cannot serve costs nothing to refuse.
        cls._check_layers(settings)
        import sglang as sgl
        from transformers import AutoTokenizer

        hidden_size = cls._hidden_size(settings.model_id, settings.revision)
        engine = sgl.Engine(
            model_path=settings.model_id,
            revision=settings.revision,
            mem_fraction_static=settings.mem_fraction_static,
            # Extraction has to be enabled when the engine starts; it cannot be
            # asked for per request afterwards.
            enable_return_hidden_states=bool(settings.feature_layers),
            **settings.engine_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
        return cls(engine, tokenizer, settings, hidden_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    def _codes_for(self, schema):
        width = max(len(field.choices) for field in schema.values())
        return candidate_codes(self.tokenizer, width)

    def _render(self, state, schema, names=None, thinking=False):
        """One prompt per question: the shared state, then that question alone.

        Questions are isolated from one another, as the schema requires, and the
        state they share is a common prefix, which is what the engine's prefix
        cache is for.
        """
        codes = self._codes_for(schema)
        prompts, labels = [], []
        for name, question in schema.items():
            if names is not None and name not in names:
                continue
            own = codes[: len(question.choices)]
            builder = build_thinking_messages if thinking else build_decision_messages
            messages = (builder(state, question_suffix(question, own)) if thinking
                        else builder(state))
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            if not thinking:
                text += question_suffix(question, own)
            prompts.append(text)
            labels.append((name, own))
        return prompts, labels

    def _candidate_ids(self, codes):
        ids = []
        for code in codes:
            pieces = self.tokenizer.encode(code, add_special_tokens=False)
            if len(pieces) != 1:
                raise ValueError(f"Candidate code {code!r} is not one token for this tokenizer")
            ids.append(pieces[0])
        return ids

    def _generate(self, prompts, max_new_tokens, candidate_ids_per_prompt):
        """Logprobs for the candidate codes, and the hidden state beside them."""
        want_hidden = bool(self.settings.feature_layers)
        results = []
        groups = ([list(range(len(prompts)))] if self.settings.batch_questions
                  else [[i] for i in range(len(prompts))])
        for group in groups:
            outputs = self.engine.generate(
                [prompts[i] for i in group],
                sampling_params=[{"max_new_tokens": max_new_tokens, "temperature": 0.0}
                                 for _ in group],
                return_logprob=True,
                token_ids_logprob=[candidate_ids_per_prompt[i] for i in group],
                return_hidden_states=want_hidden,
            )
            outputs = outputs if isinstance(outputs, list) else [outputs]
            results.extend(zip(group, outputs))
        return [output for _, output in sorted(results, key=lambda pair: pair[0])]

    @staticmethod
    def _logits_from(output, candidate_ids):
        """The candidate logprobs at the readout position, ordered as the codes are.

        The engine reports these under `output_token_ids_logprobs`, one entry per
        generated position, each a list of (logprob, token_id, text) for the ids
        that were asked for. With one generated token there is one entry, and it is
        the distribution the boundary set up, which is what the fast mode reads.

        `input_token_ids_logprobs` exists alongside it and is `[None]` unless
        `logprob_start_len` was set. Reading that one instead is how this crashed
        on the first step of the first episode, so the entries are searched for the
        first that holds anything rather than indexed by position.

        Order is by token id, not by the position the engine returned them in.
        """
        meta = output["meta_info"]
        rows = [row for row in (meta.get("output_token_ids_logprobs") or []) if row]
        if not rows:
            rows = [row for row in (meta.get("input_token_ids_logprobs") or []) if row]
        if not rows:
            raise RuntimeError(
                "SGLang returned no token-id logprobs. Both output_token_ids_logprobs "
                "and input_token_ids_logprobs were empty or all None; was "
                "token_ids_logprob passed, and was at least one token generated?")
        table = {int(token): float(value) for value, token, *_ in rows[0]}
        missing = [t for t in candidate_ids if t not in table]
        if missing:
            raise RuntimeError(f"SGLang omitted logprobs for token ids {missing}")
        return np.array([table[t] for t in candidate_ids], dtype=np.float32)

    def _hidden_from(self, output):
        """The readout position's vector, stacked as the head expects."""
        if not self.settings.feature_layers:
            return None
        return readout_hidden_state(output["meta_info"]["hidden_states"],
                                    self.settings.feature_layers)[None, :]

    def _fields(self, prompts, labels, outputs, generated):
        fields = []
        for (name, codes), prompt, output in zip(labels, prompts, outputs):
            fields.append(RawFieldScores(
                name=name,
                logits=self._logits_from(output, self._candidate_ids(codes)),
                input_tokens=int(output["meta_info"].get("prompt_tokens", 0)),
                hidden=self._hidden_from(output),
                generated_tokens=int(output["meta_info"].get("completion_tokens", 0)) if generated
                else 0,
                provenance={
                    "module": "sglang",
                    "slot_format": SLOT_FORMAT,
                    "boundary": ANSWER_BOUNDARY.strip(),
                    "codes": list(codes),
                    "batched": self.settings.batch_questions,
                },
            ))
        return tuple(fields)

    def score(self, state, schema):
        """The fast mode: no generated token, the answer read at the boundary."""
        prompts, labels = self._render(state, schema, thinking=False)
        candidate_ids = [self._candidate_ids(codes) for _, codes in labels]
        outputs = self._generate(prompts, max_new_tokens=1, candidate_ids_per_prompt=candidate_ids)
        return self._fields(prompts, labels, outputs, generated=False)

    def score_options(self, state, question, content_free=True):
        """Score each option by its own text, with no label between the two.

        The same readout as `TransformersScorer.score_options`, returning the same
        `(scores, rewritten)`, so the engine does not know which backend is under
        it. Nothing is sampled: a row is the prefix followed by the option's own
        tokens, and the engine is asked what it gave tokens it was handed rather
        than what it would have chosen.

        The scores will not equal the transformers ones and are not interchangeable
        with them. The two implementations' readout states agree to cosine 0.9995
        and differ by about 3% in magnitude, and a score here is a sum over several
        such positions, so a table may not mix the backends without saying which
        produced each row.
        """
        scored, rewritten = self._score_options(state, question)
        if not content_free:
            return scored, rewritten
        blank = question.model_copy(update={"instructions": CONTENT_FREE_INSTRUCTIONS})
        priors, _ = self._score_options(state, blank)
        return [replace(score, prior=prior.total)
                for score, prior in zip(scored, priors, strict=True)], rewritten

    def _score_options(self, state, question):
        prefix = self.tokenizer.apply_chat_template(
            build_content_messages(state), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) + unlabelled_suffix(question)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        # The same rewriting the listing does, from the same function, because the
        # two disagreeing about an option's text is the fault being avoided.
        options, rewritten = listable_options(question)
        # The separating space is part of the option, not of the prompt: BPE binds
        # it to the following word, so a prompt ending in one cannot be a prefix of
        # the joint encoding.
        per_option = [option_token_ids(self.tokenizer, prefix, " " + text)
                      for text in options]
        outputs = self._teacher_force([prefix_ids + ids for ids in per_option],
                                      len(prefix_ids))
        return [
            OptionScore(index=index, text=text, tokens=len(ids),
                        total=self._option_total(output, ids))
            for index, (text, ids, output)
            in enumerate(zip(options, per_option, outputs, strict=True))
        ], rewritten

    def _teacher_force(self, rows, prefix_length):
        """One reply per row, carrying the log-probability of each given token.

        Sent as token ids, not as text. Re-tokenising `prefix + option` on the far
        side could merge across the boundary that `option_token_ids` has just
        proved separable, and then the tokens whose log-probabilities come back
        are not the tokens whose count the normalisations divide by.

        Grouped by `batch_questions` for the reason that setting exists. SGLang's
        open misattribution bugs are about what a request gets back when it shares
        a forward pass, and here an option is a request exactly as a question is.
        The rows share their whole prefix, so the engine's prefix cache pays for
        running them one at a time.
        """
        groups = ([list(range(len(rows)))] if self.settings.batch_questions
                  else [[index] for index in range(len(rows))])
        results = []
        for group in groups:
            outputs = self.engine.generate(
                input_ids=[rows[index] for index in group],
                sampling_params=[{"max_new_tokens": 1, "temperature": 0.0}
                                 for _ in group],
                return_logprob=True,
                # One position before the first option token: the entry for a token
                # is the distribution it was drawn from, which sits at the position
                # before it. Asking from here rather than from `prefix_length` keeps
                # the request right under either reading of the parameter, and
                # `_option_total` checks the ids rather than trusting the offset.
                logprob_start_len=[prefix_length - 1] * len(group),
            )
            outputs = outputs if isinstance(outputs, list) else [outputs]
            results.extend(zip(group, outputs))
        return [output for _, output in sorted(results, key=lambda pair: pair[0])]

    @staticmethod
    def _option_total(output, option_ids):
        """The summed log-probability of this option's own tokens.

        Found by the ids that came back, not by an offset into the reply.
        `logprob_start_len` is the one part of this whose meaning has moved between
        SGLang versions, and an off-by-one there scores the last prefix token
        instead of the option's last: a number that looks ordinary, is wrong by one
        term in every option, and leaves the ranking intact so that nothing
        downstream reports it.
        """
        meta = output["meta_info"]
        entries = [entry for entry in (meta.get("input_token_logprobs") or []) if entry]
        if not entries:
            raise RuntimeError(
                "SGLang returned no input_token_logprobs, so the option's own tokens "
                "were never scored. Was return_logprob passed with logprob_start_len?")
        if len(entries) < len(option_ids):
            raise RuntimeError(
                f"SGLang returned {len(entries)} input log-probabilities for an option "
                f"of {len(option_ids)} tokens, so the reply begins after the option "
                f"does and the first of its tokens is missing")
        tail = entries[-len(option_ids):]
        got = [int(entry[1]) for entry in tail]
        if got != list(option_ids):
            raise RuntimeError(
                f"the log-probabilities came back for token ids {got}, not this "
                f"option's {list(option_ids)}: the reply is not aligned with the "
                f"request, so the sum would be over the wrong positions")
        if any(entry[0] is None for entry in tail):
            raise RuntimeError(
                "SGLang returned None for one of the option's tokens, which is what it "
                "returns for the first token of a sequence; the prefix is missing")
        return float(sum(float(entry[0]) for entry in tail))

    def think(self, state, schema, names, budget):
        """The slow mode: reason up to `budget` tokens, then read the same candidates."""
        prompts, labels = self._render(state, schema, names=set(names), thinking=True)
        candidate_ids = [self._candidate_ids(codes) for _, codes in labels]
        reasoned = self._generate(prompts, max_new_tokens=budget,
                                 candidate_ids_per_prompt=candidate_ids)
        # The answer is read after the reasoning, at the same boundary, so the
        # typed answer is assembled in code and nothing is parsed from the text.
        followups, spent = [], []
        for prompt, output in zip(prompts, reasoned):
            followups.append(prompt + output["text"] + ANSWER_BOUNDARY)
            spent.append(int(output["meta_info"].get("completion_tokens", 0)))
        outputs = self._generate(followups, max_new_tokens=1,
                                 candidate_ids_per_prompt=candidate_ids)
        fields = self._fields(followups, labels, outputs, generated=False)
        return tuple(
            RawFieldScores(name=f.name, logits=f.logits, input_tokens=f.input_tokens,
                           hidden=f.hidden, generated_tokens=tokens, provenance=f.provenance)
            for f, tokens in zip(fields, spent)
        )
