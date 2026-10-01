"""The hybrid-thinking model's two modes, served by SGLang.

Implements the same surface as `litjev.backend.TransformersScorer`:

    hidden_size                              the backbone's width
    score(state, schema)                     the fast mode, one RawFieldScores per question
    think(state, schema, names, budget)      the slow mode, for the named questions

so `SchemaDecisionEngine` takes either without knowing which.
"""

from dataclasses import dataclass, field

import numpy as np

from litjev.decision import RawFieldScores
from litjev.prompting import (
    ANSWER_BOUNDARY,
    build_decision_messages,
    build_thinking_messages,
    question_suffix,
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


class SGLangScorer:
    """Fast and slow modes over one SGLang engine."""

    def __init__(self, engine, tokenizer, settings, hidden_size):
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

    @classmethod
    def load(cls, settings):
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
        """Layer `feature_layers` at the readout position, stacked as the head expects."""
        layers = self.settings.feature_layers
        if not layers:
            return None
        states = np.asarray(output["meta_info"]["hidden_states"], dtype=np.float32)
        if states.ndim == 3:                       # [position, layer, width]
            readout = states[-2] if states.shape[0] > 1 else states[-1]
            return np.stack([readout[layer] for layer in layers])
        if states.ndim == 2:                       # [position, width]; last layer only
            if tuple(layers) != (-1,):
                raise RuntimeError(
                    f"this engine returns only the last hidden layer, but the head reads "
                    f"{layers}; use a head trained on layer -1 or an engine with per-layer "
                    f"extraction")
            readout = states[-2] if states.shape[0] > 1 else states[-1]
            return readout[None, :]
        raise RuntimeError(f"unexpected hidden_states shape {states.shape}")

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
