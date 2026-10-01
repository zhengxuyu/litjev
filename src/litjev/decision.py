"""Typed answers are assembled from logits, not generated text."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from litjev.routing import (
    OUTCOME_CLASSES,
    RoutingPolicy,
    escalation_gain,
    fast_confidence,
    should_escalate,
)
from litjev.scoring import calibrated_distribution

CONFIDENCE_METHOD = "normalized_gini_concentration_v1"
HEAD_CONFIDENCE_METHOD = "decision_head_outcome_v1"


@dataclass(frozen=True)
class RawFieldScores:
    name: str
    logits: np.ndarray
    input_tokens: int
    provenance: dict = field(default_factory=dict)
    hidden: np.ndarray | None = None
    generated_tokens: int = 0


class LogitProvider(Protocol):
    def score(self, state, schema) -> tuple[RawFieldScores, ...]: ...


@dataclass(frozen=True)
class ChoiceAnswer:
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    type: Literal["score"]
    score: float
    legend: dict
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    type: Literal["noul"]
    noul: float


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int = 0


@dataclass(frozen=True)
class DecisionResponse:
    model: str
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: Usage


@dataclass(frozen=True)
class Evaluation:
    result: DecisionResponse
    diagnostics: dict


def concentration(probabilities):
    """LitJev statistic, not a reproduction of Jev's unpublished formula."""
    count = len(probabilities)
    if count == 1:
        return 1.0
    return float(np.clip((count * sum(p * p for p in probabilities) - 1) / (count - 1), 0, 1))


# "coded" scores single-token letters; "content" scores each option's own text.
# They are different quantities on different scales, which is why a head carries
# the one it was trained under.
READOUTS = ("coded", "content")
# Every question goes through the content readout unless a caller says otherwise.
# `coded` stays reachable, and the paths that reproduce published numbers name it:
# those numbers were produced by scoring letter codes and would not reproduce
# under a readout that scores option text.
DEFAULT_READOUT = "content"
# How an option's score is read off its `OptionScore`. `sum` is the default; see
# the note in SchemaDecisionEngine.__init__ for why it is not `mean`.
RANKED_BY = {"sum": "total", "mean": "mean", "pmi": "pmi"}
NORMALISATIONS = tuple(RANKED_BY)


class SchemaDecisionEngine:
    def __init__(
        self,
        provider,
        temperature=1.0,
        model_id="unknown",
        calibration_fitted=False,
        head=None,
        routing=None,
        readout=DEFAULT_READOUT,
        normalisation="sum",
    ):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.provider = provider
        self.temperature = temperature
        self.model_id = model_id
        self.calibration_fitted = calibration_fitted
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}; use one of {sorted(READOUTS)}")
        if normalisation not in NORMALISATIONS:
            raise ValueError(
                f"unknown normalisation {normalisation!r}; use one of "
                f"{sorted(NORMALISATIONS)}")
        self.readout = readout
        # Which of the option scores is ranked on, under the content readout. The
        # default is `sum`: it is the joint log-probability of the option, it was
        # the most length-neutral of the three within a question, and its
        # confidence ranked right answers above wrong ones best. Recorded with
        # every result, because two runs under different normalisations are two
        # different measurements.
        self.normalisation = normalisation
        if readout == "content" and not hasattr(provider, "score_options"):
            raise NotImplementedError(
                f"{type(provider).__name__} cannot score option content, which is what "
                f"the default readout does; pass readout='coded' to serve letter codes. "
                f"Both shipped backends implement it, so this is a provider that does "
                f"not, and it is refused here rather than at the first question")
        self.head = head
        # Refused here rather than at the first prediction. A head fitted to one
        # readout's distribution and served under the other reads numbers that do
        # not mean what it learned they meant, and its output looks ordinary.
        if head is not None:
            # The engine's model, not the head's own, which would compare a value
            # against itself and pass whatever it held. "unknown" is the default
            # and is not a claim about which model this is, so it checks the
            # readout alone rather than failing every engine built without one.
            head.metadata.check_serving(
                self.model_id if model_id != "unknown" else head.metadata.model_id,
                readout=readout)
        self.routing = routing if routing is not None else RoutingPolicy()

    def decide(self, state, schema, routing=None):
        return self.evaluate(state, schema, routing).result

    def _distribution(self, row, question):
        if len(row.logits) != len(question.choices):
            raise RuntimeError("Scorer returned mismatched candidates")
        distribution = calibrated_distribution(
            row.logits, list(range(len(question.choices))), self.temperature
        )
        probabilities = dict(zip(question.choices, distribution.probabilities, strict=True))
        return distribution, probabilities

    @staticmethod
    def _answer(question, distribution, probabilities, confidence):
        if question.type == "noul":
            return NoulAnswer("noul", probabilities["true"])
        if question.type == "score":
            return ScoreAnswer(
                "score",
                sum(int(k) * p for k, p in probabilities.items()),
                dict(zip(question.choices, question.descriptions, strict=True)),
                probabilities,
                confidence,
            )
        return ChoiceAnswer(
            "choice", question.choices[distribution.winner_index], probabilities, confidence
        )

    def _outcome(self, row, question, distribution):
        """Decision head prediction for one fast readout, or None without a head."""
        if self.head is None:
            return None
        from litjev.heads import build_features

        hidden = row.hidden
        if not self.head.metadata.feature_layers:
            hidden = np.zeros(0, dtype=np.float32)  # stats-only head
        elif hidden is None:
            raise RuntimeError("Decision head requires readout hidden states from the scorer")
        features = build_features(hidden, distribution.probabilities, question.type)
        return self.head.predict(features)[0]

    def evaluate(self, state, schema, routing=None):
        policy = routing if routing is not None else self.routing
        if policy.enabled and self.head is None:
            raise ValueError("Routing to the slow path requires a decision head")
        readout = self.readout_for(state)
        scores = self.readout_scores(state, schema)
        if tuple(row.name for row in scores) != schema.names:
            raise RuntimeError("Scorer returned mismatched fields")
        answers, fields, outcomes, escalate = {}, {}, {}, []
        for row in scores:
            question = schema[row.name]
            distribution, probabilities = self._distribution(row, question)
            outcome = self._outcome(row, question, distribution)
            if outcome is None:
                confidence = concentration(distribution.probabilities)
            else:
                confidence = fast_confidence(outcome)
                outcomes[row.name] = outcome
                if should_escalate(outcome, policy):
                    escalate.append(row.name)
            answers[row.name] = self._answer(question, distribution, probabilities, confidence)
            fields[row.name] = {
                "logits": tuple(float(value) for value in row.logits),
                "probabilities": probabilities,
                "max_probability": distribution.winner_probability,
                "concentration": concentration(distribution.probabilities),
                "system": "one",
                "provenance": {**row.provenance, "temperature": self.temperature},
            }
            if outcome is not None:
                fields[row.name]["outcome"] = dict(
                    zip(OUTCOME_CLASSES, (float(p) for p in outcome), strict=True)
                )
                fields[row.name]["escalation_gain"] = escalation_gain(outcome)
        output_tokens = 0
        forward_calls = 2
        if escalate:
            slow = self.provider.think(state, schema, tuple(escalate), policy.budget)
            if tuple(row.name for row in slow) != tuple(escalate):
                raise RuntimeError("Slow path returned mismatched fields")
            for row in slow:
                question = schema[row.name]
                distribution, probabilities = self._distribution(row, question)
                outcome = outcomes[row.name]
                slow_confidence = float(outcome[0] + outcome[2])
                answers[row.name] = self._answer(
                    question, distribution, probabilities, slow_confidence
                )
                output_tokens += row.generated_tokens
                fields[row.name].update(
                    {
                        "fast_logits": fields[row.name]["logits"],
                        "fast_probabilities": fields[row.name]["probabilities"],
                        "logits": tuple(float(value) for value in row.logits),
                        "probabilities": probabilities,
                        "max_probability": distribution.winner_probability,
                        "concentration": concentration(distribution.probabilities),
                        "system": "two",
                        "slow_provenance": {**row.provenance, "temperature": self.temperature},
                    }
                )
            # One prefill, one forward per generated token, one final readout.
            forward_calls += 2 + max(row.generated_tokens for row in slow)
        return Evaluation(
            DecisionResponse(self.model_id, answers, Usage(scores[0].input_tokens, output_tokens)),
            {
                "fields": fields,
                "forward_calls": forward_calls,
                "calibration_fitted": self.calibration_fitted,
                "confidence_method": HEAD_CONFIDENCE_METHOD if self.head else CONFIDENCE_METHOD,
                "readout": readout,
                "normalisation": self.normalisation if readout == "content" else None,
                "routing": {
                    "enabled": policy.enabled,
                    "lambda": policy.lambda_,
                    "budget": policy.budget,
                    "escalated": escalate,
                    "generated_tokens": output_tokens,
                },
            },
        )

    def readout_for(self, state):
        """The readout this state is actually read under.

        A screenshot goes through the coded readout whatever the engine is set to:
        the content prompt is text only, and the image path compiles its slots
        around the processor's own image tokens. It is returned rather than
        applied silently, so the diagnostics name the readout that ran.
        """
        from litjev.vision import VisualState

        return "coded" if isinstance(state, VisualState) else self.readout

    def readout_scores(self, state, schema):
        """The fast mode's scores, through the readout `readout_for` names.

        Public because a caller that does not serve through `evaluate` still has
        to read out the way the engine says, or it records a readout it was not
        using.
        """
        return (self._content_scores(state, schema) if self.readout_for(state) == "content"
                else self.provider.score(state, schema))

    def _content_scores(self, state, schema):
        """Read out by scoring each option's own text, with no label between them.

        Returned in the shape the coded readout returns, so everything after this
        -- the distribution, the head's features, the answer -- is the same code.
        The scores are log-probabilities rather than logits over a shared
        vocabulary, which is why a head carries the readout it was trained under:
        a softmax over these is not the quantity a coded-trained head learned.

        No hidden state: `score_options` does not produce one, and a head would
        need it. Routing under this readout therefore needs a head trained under
        it, which the check in `__init__` already requires.
        """
        rows = []
        for name in schema.names:
            found, rewritten = self.provider.score_options(
                state, schema[name], content_free=self.normalisation == "pmi")
            ranked = {score.index: getattr(score, RANKED_BY[self.normalisation])
                      for score in found}
            rows.append(RawFieldScores(
                name, np.array([ranked[i] for i in sorted(ranked)], dtype=np.float64),
                0, {"system": "one", "readout": "content",
                    "normalisation": self.normalisation,
                    "rewritten_options": rewritten}, None))
        return tuple(rows)
