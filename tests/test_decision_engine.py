import numpy as np
import pytest

from litjev.decision import RawFieldScores, SchemaDecisionEngine, concentration
from litjev.schema import Choice, DecisionSchema, Noul


class FakeLogitProvider:
    def score(self, state, schema):
        return (
            RawFieldScores("q1", np.array([3.0, 1.0]), 12),
            RawFieldScores("q2", np.array([0.0, 4.0]), 12),
        )


def test_typed_results_and_separate_diagnostics():
    schema = DecisionSchema({"q1": Choice(criteria={"A": None, "B": None}), "q2": Noul()})
    engine = SchemaDecisionEngine(FakeLogitProvider(), temperature=2, readout="coded")
    evaluation = engine.evaluate("state", schema)
    response = evaluation.result
    assert response.answers["q1"].choice == "A"
    assert response.answers["q2"].noul == pytest.approx(0.8807970779)
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 0
    assert evaluation.diagnostics["fields"]["q1"]["max_probability"] < 0.8
    assert concentration([0.5, 0.5]) == 0
    assert concentration([1.0, 0.0]) == 1
