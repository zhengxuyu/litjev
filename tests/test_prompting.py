from test_visual import ImageTokenizer

from litjev.prompting import build_decision_messages
from litjev.schema import Choice, DecisionSchema
from litjev.slots import compile_slots


def test_question_is_user_content_and_answer_is_assistant_content():
    compiled = compile_slots(ImageTokenizer(), "scene", DecisionSchema({
        "action": Choice(instructions="Choose", criteria={"left": None, "right": None})
    }))
    assert "<assistant>" not in compiled.prefix_text
    suffix = compiled.slot_texts[0]
    assert suffix.startswith("<user>Question:")
    assert suffix.endswith("</end><assistant>Answer:")
    assert suffix.index("Question:") < suffix.index("<assistant>")


def test_question_ids_and_other_questions_never_enter_branch():
    schema = DecisionSchema(
        {
            f"SECRET_ID_{i}": Choice(
                instructions=f"unique question {i}", criteria={"long option": "yes", "B": "no"}
            )
            for i in range(10)
        }
    )
    messages = build_decision_messages({"shared": [1, 2]}, schema)
    assert messages[1]["content"] == '{"shared": [1, 2]}'
    compiled = compile_slots(ImageTokenizer(), "state", schema)
    assert all("SECRET_ID" not in text for text in compiled.slot_texts)
    for i, text in enumerate(compiled.slot_texts):
        for j in range(10):
            assert (f"unique question {j}" in text) == (i == j)
    renamed = DecisionSchema({str(i): q for i, q in enumerate(schema.values())})
    assert compile_slots(ImageTokenizer(), "state", renamed).input_ids == compiled.input_ids
    assert compiled.candidates[0] == [ord("A") + 2, ord("B") + 2]
