from dataclasses import asdict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from litjev.api import create_app
from litjev.decision import RawFieldScores, SchemaDecisionEngine
from litjev.schema import DecisionSchema


class Provider:
    def score(self, state, schema):
        return tuple(RawFieldScores(k, np.arange(len(v.choices)), 42) for k, v in schema.items())


def test_standard_and_debug_share_canonical_answers():
    engine = SchemaDecisionEngine(Provider(), model_id="test-qwen", readout="coded")
    client = TestClient(create_app(lambda: engine))
    questions = {
        "pick": {
            "type": "choice",
            "instructions": "Pick",
            "criteria": {"long key": None, "B": "two"},
        },
        "rate": {"type": "score", "instructions": {"task": "Rate"}, "criteria": [None, {"x": 2}]},
        "yes": {"type": "noul", "instructions": "Is it?"},
    }
    payload = {"model": "litjev", "state": [{"value": 3}], "questions": questions}
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"model", "answers", "usage"}
    assert result["model"] == "test-qwen"
    assert result["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert set(result["answers"]["pick"]) == {"type", "choice", "probabilities", "confidence"}
    assert result["answers"]["pick"]["choice"] == "B"
    assert result["answers"]["rate"]["legend"] == {"0": None, "1": {"x": 2}}
    assert result["answers"]["rate"]["score"] == pytest.approx(0.7310585786)
    assert result["answers"]["yes"] == {"type": "noul", "noul": pytest.approx(0.7310585786)}
    debug = client.post("/v1/systemone/debug", json=payload).json()
    assert debug["result"] == result
    assert debug["diagnostics"]["forward_calls"] == 2
    assert "timing" in debug["diagnostics"]
    assert asdict(engine.decide(payload["state"], DecisionSchema.from_mapping(questions))) == result


@pytest.mark.parametrize(
    "question",
    [
        {"type": "enum", "description": "Old", "choices": ["A", "B"]},
        {"type": "choice", "instructions": "Pick", "criteria": {}},
        {"type": "score", "instructions": "Rate", "criteria": ["one"]},
        {"type": "noul", "instructions": "Yes?", "criteria": {"maybe": "invalid"}},
        {"type": "choice", "instructions": "Pick", "criteria": {"A": 3}},
    ],
)
def test_invalid_questions_fail_before_model_load(question):
    def forbidden():
        raise AssertionError("invalid request must not load weights")

    client = TestClient(create_app(forbidden))
    response = client.post(
        "/v1/systemone", json={"model": "litjev", "state": "", "questions": {"q": question}}
    )
    assert response.status_code == 422


def test_ten_questions_in_one_call_and_unknown_model_rejected():
    calls = []

    class CountingProvider(Provider):
        def score(self, state, schema):
            calls.append(len(schema))
            return super().score(state, schema)

    client = TestClient(
        create_app(lambda: SchemaDecisionEngine(CountingProvider(), model_id="qwen", readout="coded"))
    )
    payload = {
        "model": "litjev",
        "state": "",
        "questions": {
            f"q{i}": {"type": "choice", "instructions": "Pick", "criteria": {"A": "1", "B": "2"}}
            for i in range(10)
        },
    }
    assert len(client.post("/v1/systemone", json=payload).json()["answers"]) == 10
    assert calls == [10]
    payload["model"] = "not-loaded"
    assert client.post("/v1/systemone", json=payload).status_code == 422
    assert calls == [10]
