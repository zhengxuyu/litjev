import numpy as np
from fastapi.testclient import TestClient

from litjev.api import create_app
from litjev.decision import RawFieldScores, SchemaDecisionEngine


class FakeProvider:
    def score(self, state, schema):
        return tuple(RawFieldScores(k, np.zeros(len(v.choices)), 42) for k, v in schema.items())


class FakeEngine(SchemaDecisionEngine):
    def __init__(self):
        # Coded: FakeProvider implements `score` and not `score_options`, which is
        # what these tests are about -- the API, not the readout.
        super().__init__(FakeProvider(), model_id="fake", readout="coded")


def test_playground_serves_without_loading_model():
    def forbidden_load():
        raise AssertionError("Opening the UI must not load weights")

    client = TestClient(create_app(forbidden_load))
    assert client.get("/").status_code == 200
    assert "schema-input" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert not client.get("/health").json()["model_loaded"]


def test_legacy_endpoints_are_not_silently_accepted():
    client = TestClient(create_app(FakeEngine))
    for endpoint in ["/v1/batch-mcq", "/v1/calibrated-schema"]:
        assert client.post(endpoint, json={}).status_code == 404
