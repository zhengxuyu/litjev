import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import LlamaConfig, LlamaForCausalLM

from litjev.api import create_app
from litjev.backend import SuffixStop, TransformersScorer
from litjev.collect import collect_batch, load_records, save_records
from litjev.decision import HEAD_CONFIDENCE_METHOD, RawFieldScores, SchemaDecisionEngine
from litjev.heads import (
    STATS_DIM,
    DecisionHead,
    HeadMetadata,
    auroc,
    build_features,
    coverage_accuracy_curve,
    distribution_stats,
    outcome_labels,
    train_head,
)
from litjev.prompting import ANSWER_BOUNDARY, build_thinking_messages, question_body
from litjev.routing import RoutingPolicy, escalation_gain, fast_confidence, should_escalate
from litjev.schema import Choice, DecisionSchema, Noul, SystemOneRequest
from litjev.training import run_training


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        # Deterministic stand-in: content joined, with a visible marker for thinking mode.
        rendered = "|".join(m["content"] for m in messages)
        return rendered + ("<think>\n" if kwargs.get("enable_thinking") else "|")

    def encode(self, text, **kwargs):
        return [ord(char) + 2 for char in text.replace(": ", ":")]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 2) for i in ids)


def tiny_model(seed=4):
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()


def two_questions():
    return DecisionSchema(
        {
            "q1": Choice(instructions="Pick", criteria={"A": None, "B": None, "C": None}),
            "q2": Noul(),
        }
    )


# ---------------------------------------------------------------- head & routing


def test_stats_features_and_outcome_labels():
    stats = distribution_stats([0.7, 0.2, 0.1], "choice")
    assert stats.shape == (STATS_DIM,)
    assert stats[0] == pytest.approx(0.7)
    assert stats[2] == pytest.approx(0.5)
    assert list(stats[5:]) == [1.0, 0.0, 0.0]
    assert build_features(np.zeros((2, 4)), [0.5, 0.5], "noul").shape == (8 + STATS_DIM,)
    with pytest.raises(ValueError):
        distribution_stats([1.0], "essay")
    labels = outcome_labels([True, True, False, False], [True, False, True, False])
    assert labels.tolist() == [0, 1, 2, 3]


def test_routing_gain_and_confidence():
    outcome = np.array([0.5, 0.1, 0.3, 0.1])
    assert fast_confidence(outcome) == pytest.approx(0.6)
    assert escalation_gain(outcome) == pytest.approx(0.2)
    assert should_escalate(outcome, RoutingPolicy(0.1, budget=16))
    assert not should_escalate(outcome, RoutingPolicy(0.25, budget=16))
    assert not should_escalate(outcome, RoutingPolicy(0.0, budget=0))
    with pytest.raises(ValueError):
        RoutingPolicy(float("nan"))


def test_head_roundtrip_and_serving_checks(tmp_path):
    meta = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1, 3), hidden_width=8)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, meta.input_dim)).astype(np.float32)
    labels = rng.integers(0, 4, size=64)
    head, history = train_head(meta, features, labels, epochs=2, batch_size=16)
    assert len(history) == 2
    assert {"train_loss", "val_loss"} <= set(history[0])
    probs = head.predict(features[:3])
    assert probs.shape == (3, 4)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)
    path = tmp_path / "head.safetensors"
    head.save(path)
    loaded = DecisionHead.load(path)
    assert loaded.metadata == meta
    np.testing.assert_allclose(loaded.predict(features[:3]), probs, atol=1e-6)
    loaded.metadata.check_serving("m", "r")
    with pytest.raises(ValueError):
        loaded.metadata.check_serving("other")
    # PCA bottleneck: whitened components replace the raw hidden block, stats pass through.
    pca_meta = HeadMetadata(
        "m", "r", hidden_size=4, feature_layers=(-1, 3), pca_dim=3, hidden_width=8
    )
    pca_head, _ = train_head(pca_meta, features, labels, epochs=2, batch_size=16)
    assert pca_head.projection.shape == (8, 3)
    assert pca_meta.model_dim == 3 + STATS_DIM
    pca_head.save(tmp_path / "pca.safetensors")
    reloaded = DecisionHead.load(tmp_path / "pca.safetensors")
    np.testing.assert_allclose(
        reloaded.predict(features[:2]), pca_head.predict(features[:2]), atol=1e-6
    )
    with pytest.raises(ValueError):
        DecisionHead(HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,), pca_dim=9))
    with pytest.raises(ValueError):
        loaded.metadata.check_serving("m", "different-revision")


def test_auroc_and_coverage_curve():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [True, True]))
    curve = coverage_accuracy_curve(
        gain=[0.4, -0.3, 0.1],
        fast_correct=[False, True, True],
        slow_correct=[True, False, True],
        lambdas=[0.0, 0.2, 1.0],
    )
    assert [row["escalation_rate"] for row in curve] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert [row["accuracy"] for row in curve] == pytest.approx([1.0, 1.0, 2 / 3])


# ---------------------------------------------------------------- backend


def test_fast_path_returns_readout_hidden_states():
    model = tiny_model()
    scorer = TransformersScorer(model, CharTokenizer(), feature_layers=(-1, 1))
    schema = two_questions()
    scores = scorer.score("state", schema)
    compiled = scorer._compile("state", schema)
    assert scorer.hidden_size == 32
    with torch.inference_mode():
        for i, row in enumerate(compiled.input_ids):
            assert scores[i].hidden.shape == (2, 32)
            full = model(torch.tensor([row]), use_cache=False, output_hidden_states=True)
            np.testing.assert_allclose(
                scores[i].hidden[0], full.hidden_states[-1][0, -1].numpy(), atol=1e-5
            )
            np.testing.assert_allclose(
                scores[i].hidden[1], full.hidden_states[1][0, -1].numpy(), atol=1e-5
            )
    assert scores[0].provenance["system"] == "one"
    assert scores[0].provenance["feature_layers"] == [-1, 1]


def test_slow_path_reads_same_boundary_after_thinking():
    model = tiny_model()
    tokenizer = CharTokenizer()
    scorer = TransformersScorer(model, tokenizer, feature_layers=(-1,))
    schema = two_questions()
    compiled = scorer._compile("state", schema)
    slow = scorer.think("state", schema, ("q2", "q1"), budget=5)
    assert tuple(row.name for row in slow) == ("q2", "q1")
    with torch.inference_mode():
        for row in slow:
            i = schema.names.index(row.name)
            assert row.provenance["system"] == "two"
            assert 0 <= row.generated_tokens <= 5
            thought = tokenizer.encode(row.provenance["thinking_text"])
            assert len(thought) == row.generated_tokens
            body = question_body(schema[row.name], compiled.candidate_codes[i])
            prompt = tokenizer.apply_chat_template(
                build_thinking_messages("state", body), enable_thinking=True
            )
            assert prompt.endswith("<think>\n") and "state\n\n" + body in prompt
            sequence = (
                tokenizer.encode(prompt)
                + thought
                + tokenizer.encode("\n</think>\n" + ANSWER_BOUNDARY)
            )
            assert row.input_tokens == len(sequence)
            full = model(torch.tensor([sequence]), use_cache=False, output_hidden_states=True)
            np.testing.assert_allclose(
                row.logits, full.logits[0, -1, compiled.candidates[i]].numpy(), atol=1e-5
            )
            np.testing.assert_allclose(
                row.hidden[0], full.hidden_states[-1][0, -1].numpy(), atol=1e-5
            )
    with pytest.raises(ValueError):
        scorer.think("state", schema, ("missing",), budget=5)
    with pytest.raises(ValueError):
        scorer.think("state", schema, ("q1",), budget=0)


def test_slow_path_strips_closing_delimiter_and_padding(monkeypatch):
    model = tiny_model()
    tokenizer = CharTokenizer()
    scorer = TransformersScorer(model, tokenizer)
    schema = two_questions()
    close = tokenizer.encode("</think>")
    body = tokenizer.encode("xy")

    def fake_generate(input_ids, attention_mask, **kwargs):
        width = input_ids.shape[1]
        tail = torch.tensor([body + close + [0, 0]] * input_ids.shape[0])
        return torch.cat([input_ids, tail], dim=1)[:, : width + len(body) + len(close) + 2]

    monkeypatch.setattr(scorer.model, "generate", fake_generate)
    (row,) = scorer.think("state", schema, ("q1",), budget=8)
    assert row.generated_tokens == 2
    assert row.provenance["thinking_text"] == "xy"


def test_suffix_stop_is_per_row():
    stop = SuffixStop([7, 8], prompt_width=2)
    ids = torch.tensor([[1, 1, 7, 8], [1, 1, 8, 7], [1, 1, 0, 0]])
    assert stop(ids, None).tolist() == [True, False, False]
    assert stop(torch.tensor([[1, 1, 7]]), None).tolist() == [False]


# ---------------------------------------------------------------- engine


class FakeProvider:
    def __init__(self):
        self.think_calls = []

    def score(self, state, schema):
        return (
            RawFieldScores("q1", np.array([2.0, 1.0, 0.0]), 20, {"system": "one"}, np.ones((1, 4))),
            RawFieldScores("q2", np.array([0.0, 1.0]), 20, {"system": "one"}, np.ones((1, 4))),
        )

    def think(self, state, schema, names, budget):
        self.think_calls.append((names, budget))
        return tuple(
            RawFieldScores(
                name,
                np.array([0.0, 0.0, 3.0]) if name == "q1" else np.array([3.0, 0.0]),
                40,
                {"system": "two", "thinking_text": "hmm"},
                np.ones((1, 4)),
                generated_tokens=7,
            )
            for name in names
        )


class FakeHead:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.metadata = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,))

    def predict(self, features):
        return np.array([self.outcomes.pop(0)])


def test_engine_escalates_only_when_gain_exceeds_lambda():
    provider = FakeProvider()
    head = FakeHead([[0.2, 0.1, 0.6, 0.1], [0.8, 0.1, 0.05, 0.05]])
    engine = SchemaDecisionEngine(provider, head=head, routing=RoutingPolicy(0.2, budget=32),
                                    readout="coded")
    schema = two_questions()
    evaluation = engine.evaluate("state", schema)
    result = evaluation.result
    assert provider.think_calls == [(("q1",), 32)]
    assert result.answers["q1"].choice == "C"
    assert result.answers["q1"].confidence == pytest.approx(0.8)
    assert result.answers["q2"].noul == pytest.approx(1 / (1 + np.exp(-1)))
    assert result.usage.output_tokens == 7
    fields = evaluation.diagnostics["fields"]
    assert fields["q1"]["system"] == "two"
    assert fields["q1"]["fast_probabilities"]["A"] > 0.6
    assert fields["q1"]["escalation_gain"] == pytest.approx(0.5)
    assert fields["q2"]["system"] == "one"
    assert evaluation.diagnostics["routing"]["escalated"] == ["q1"]
    assert evaluation.diagnostics["forward_calls"] == 2 + 2 + 7
    assert evaluation.diagnostics["confidence_method"] == HEAD_CONFIDENCE_METHOD


def test_engine_without_routing_never_thinks_and_head_needs_hidden():
    provider = FakeProvider()
    head = FakeHead([[0.2, 0.1, 0.6, 0.1], [0.8, 0.1, 0.05, 0.05]])
    engine = SchemaDecisionEngine(provider, head=head, readout="coded")
    evaluation = engine.evaluate("state", two_questions())
    assert provider.think_calls == []
    assert evaluation.result.answers["q1"].confidence == pytest.approx(0.3)
    assert evaluation.result.usage.output_tokens == 0
    assert evaluation.diagnostics["forward_calls"] == 2
    with pytest.raises(ValueError):
        SchemaDecisionEngine(provider, routing=RoutingPolicy(0.0, 8), readout="coded").evaluate("s", two_questions())

    class NoHidden(FakeProvider):
        def score(self, state, schema):
            return tuple(
                RawFieldScores(row.name, row.logits, row.input_tokens)
                for row in super().score(state, schema)
            )

    with pytest.raises(RuntimeError):
        SchemaDecisionEngine(NoHidden(), head=FakeHead([[1, 0, 0, 0]]), readout="coded").evaluate(
            "s", two_questions()
        )

    class StatsOnlyHead(FakeHead):
        def __init__(self, outcomes):
            super().__init__(outcomes)
            self.metadata = HeadMetadata("m", "r", hidden_size=4, feature_layers=())

    # A stats-only head needs no hidden states from the scorer at all.
    stats_engine = SchemaDecisionEngine(
        NoHidden(), head=StatsOnlyHead([[0.7, 0.1, 0.1, 0.1], [0.2, 0.2, 0.5, 0.1]]),
        readout="coded")
    result = stats_engine.evaluate("s", two_questions()).result
    assert result.answers["q1"].confidence == pytest.approx(0.8)


# ---------------------------------------------------------------- collection & training


def test_collect_save_load_and_train(tmp_path):
    model = tiny_model()
    scorer = TransformersScorer(model, CharTokenizer(), feature_layers=(-1, 1))
    request = SystemOneRequest(
        model="litjev",
        state="Answer.",
        questions={
            f"q{i}": Choice(instructions=f"Pick {i}", criteria={"A": "1", "B": "2"})
            for i in range(6)
        },
    )
    labels = {f"q{i}": "A" for i in range(6)}
    categories = {f"q{i}": "math" if i % 2 else "law" for i in range(6)}
    records = collect_batch(scorer, request, list(labels), labels, 3, categories)
    assert len(records) == 6
    assert records[0].hidden.shape == (2, 32)
    assert records[0].stats.shape == (STATS_DIM,)
    metadata = {
        "model_id": "tiny",
        "revision": "r",
        "hidden_size": 32,
        "feature_layers": [-1, 1],
        "budget": 3,
    }
    path = tmp_path / "records.npz"
    save_records(path, records, metadata)
    loaded, meta = load_records([path, path])
    assert loaded["hidden"].shape == (12, 2, 32)
    assert meta["hidden_size"] == 32
    assert meta["count"] == 6
    with pytest.raises(ValueError):
        save_records(tmp_path / "empty.npz", [], metadata)

    rng = np.random.default_rng(0)
    loaded["fast_correct"] = rng.random(12) < 0.6
    loaded["slow_correct"] = rng.random(12) < 0.7
    head, report = run_training(
        loaded, meta, holdout_fraction=0.5, epochs=2, probe_epochs=1, selection_folds=2, pca_dim=2
    )
    assert report["split_level"] == "category"
    assert report["categories_seen"] == ["law", "math"]
    names = [c["name"] for c in report["candidates"]]
    assert names == ["stats_only", "layer_-1", "layer_-1_pca2", "layer_1", "layer_1_pca2"]
    assert report["chosen"] in names
    assert all({"validation", "test", "selection_score"} <= set(c) for c in report["candidates"])
    assert all(c["validation"]["folds"] == 2 for c in report["candidates"])
    # A single-category collection cannot hold out a category; fall back to examples.
    single = {**loaded, "category": np.array(["law"] * 12)}
    _, fallback = run_training(
        single, meta, holdout_fraction=0.5, epochs=1, probe_epochs=1, selection_folds=2, pca_dim=2
    )
    assert fallback["split_level"] == "example"
    assert 0 < fallback["test_count"] < 12
    assert 0 < report["train_count"] < 12
    assert set(report["head"]) == {"auroc_fast_correct", "auroc_gain_vs_helps", "curve"}
    assert head.metadata.training["chosen_layers"] == report["chosen_layers"]
    out = tmp_path / "head.safetensors"
    head.save(out)
    assert DecisionHead.load(out).metadata.feature_layers == tuple(report["chosen_layers"])
    json.dumps(report)


# ---------------------------------------------------------------- API


def test_systemtwo_endpoint_passes_routing_and_keeps_standard_shape():
    from litjev.decision import ChoiceAnswer, DecisionResponse, Evaluation, Usage

    class RecordingEngine:
        model_id = "litjev"

        def __init__(self):
            self.calls = []

        def evaluate(self, state, schema, routing=None):
            self.calls.append(routing)
            answer = ChoiceAnswer("choice", "A", {"A": 0.9, "B": 0.1}, 0.9)
            return Evaluation(
                DecisionResponse("tiny", {"q1": answer}, Usage(10, 5 if routing else 0)),
                {"fields": {}, "forward_calls": 2, "calibration_fitted": False},
            )

    engine = RecordingEngine()
    client = TestClient(create_app(lambda: engine))
    body = {
        "model": "litjev",
        "state": "s",
        "questions": {
            "q1": {"type": "choice", "instructions": "?", "criteria": {"A": "1", "B": "2"}}
        },
    }
    plain = client.post("/v1/systemtwo", json=body)
    assert plain.status_code == 200
    assert set(plain.json()) == {"model", "answers", "usage"}
    assert engine.calls[-1] is None
    routed = client.post("/v1/systemtwo", json={**body, "routing": {"lambda": 0.1, "budget": 64}})
    assert routed.status_code == 200
    assert routed.json()["usage"]["output_tokens"] == 5
    assert engine.calls[-1] == RoutingPolicy(0.1, 64)
    assert client.post("/v1/systemone", json={**body, "routing": {}}).status_code == 422
    assert client.post("/v1/systemtwo", json={**body, "routing": {"budget": 0}}).status_code == 422
    debug = client.post("/v1/systemtwo/debug", json={**body, "routing": {"lambda": 0.0}})
    assert debug.status_code == 200
    assert engine.calls[-1] == RoutingPolicy(0.0, 512)
