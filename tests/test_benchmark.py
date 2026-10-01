import numpy as np

from litjev.benchmark import evaluate
from litjev.decision import RawFieldScores, SchemaDecisionEngine
from litjev.schema import Choice, DecisionSchema


def test_sequential_uses_ten_separate_single_question_calls():
    calls = []

    class Provider:
        def score(self, state, schema):
            calls.append(schema.names)
            return tuple(RawFieldScores(k, np.zeros(2), 10) for k in schema)

    engine = SchemaDecisionEngine(Provider(), readout="coded")
    schema = DecisionSchema(
        {
            f"q{i}": Choice(instructions=f"Question {i}", criteria={"A": None, "B": None})
            for i in range(10)
        }
    )
    evaluation, durations = evaluate(engine, schema, sequential=True)
    assert len(calls) == 10 and all(len(call) == 1 for call in calls)
    assert evaluation.diagnostics["forward_calls"] == 20
    assert len(evaluation.result.answers) == len(durations) == 10
    calls.clear()
    evaluation, durations = evaluate(engine, schema)
    assert len(calls) == 1 and len(calls[0]) == 10
    assert evaluation.diagnostics["forward_calls"] == 2
    assert durations == {}
