import json


def state_text(state):
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
    )


def build_decision_messages(state, schema=None):
    """Only shared state is prefetched; question ids never affect inference."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and labeled options that follow. "
                "Return only the option code. Do not explain or reason aloud."
            ),
        },
        {"role": "user", "content": state_text(state)},
    ]


def question_suffix(question, codes):
    options = [
        {"code": codes[i], "option": key, "description": description}
        for i, (key, description) in enumerate(
            zip(question.choices, question.descriptions, strict=True)
        )
    ]
    body = {"type": question.type, "instructions": question.instructions, "options": options}
    return "Question: " + json.dumps(body, ensure_ascii=False, allow_nan=False)


def render_question_branch(tokenizer, question, codes):
    """Open a user question after the closed shared-state turn, then an answer turn."""
    history = build_decision_messages("")
    prefix = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        history + [{"role": "user", "content": question_suffix(question, codes)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not full.startswith(prefix):
        raise ValueError("Chat template does not support an append-only user question branch")
    return full[len(prefix):] + "Answer:"
