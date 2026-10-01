import json
import re


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


def build_thinking_messages(state, body):
    """Slow path: the question joins the state in the user turn and reasoning is allowed."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and labeled options. "
                "Think it through first, then answer with only the option code."
            ),
        },
        {"role": "user", "content": state_text(state) + "\n\n" + body},
    ]


ANSWER_BOUNDARY = "\nAnswer:"


def question_body(question, codes):
    """The question text shared by the fast readout and the slow thinking path."""
    options = [
        {"code": codes[i], "option": key, "description": description}
        for i, (key, description) in enumerate(
            zip(question.choices, question.descriptions, strict=True)
        )
    ]
    body = {"type": question.type, "instructions": question.instructions, "options": options}
    return "Question: " + json.dumps(body, ensure_ascii=False, allow_nan=False)


LINE_BREAK = re.compile(r"[\r\n]+")


def listable_options(question):
    """`(texts, rewritten)`: the option texts as one line each, and how many moved.

    A line break inside an option would be read as the start of the next one, so
    the prompt would show more options than are being scored. Refusing was the
    first answer and is the wrong one here: once every question goes through this
    readout, a refusal is a question that cannot be answered at all, and the
    MMLU-Pro converter keeps a source newline wherever the benchmark has one.

    So the break becomes a space -- in the listing and in the scoring alike,
    which is the whole point, since the two disagreeing is the fault being
    avoided -- and the count goes in the run's record. A prompt quietly unlike
    the benchmark is worth knowing about even when it is the right thing to do.
    """
    texts, rewritten = [], 0
    for text in question.descriptions:
        flat = LINE_BREAK.sub(" ", text) if isinstance(text, str) else text
        rewritten += flat != text
        texts.append(flat)
    return texts, rewritten


def unlabelled_body(question):
    """The question and its options, with nothing standing in for an option.

    The labelled form gives every option a single-token code and the readout
    scores those codes, which is why the readout has a preference over positions
    at all: a code is a token, a token has a prior, and the prior is attached to
    whatever sits in that slot. Here the options are a list of their own content
    and the readout scores the content instead, so there is no label to prefer.

    The order still exists, so listing order can still bias the answer. That is
    measured rather than assumed away.

    Plain lines rather than JSON, because the readout scores the option's own
    text and the prompt has to show that same text. Through `json.dumps` the
    model reads `"\\frac{1}{2}"`, escaped and in quotes, while the score is for
    `\frac{1}{2}` -- so the system prompt asks for one of the options exactly as
    written and the thing being scored is not what was written.

    One line per option, so an option carrying a newline would be read as two and
    the listing would disagree with the scoring about how many options there are.
    `listable_options` rewrites those breaks to spaces, identically here and in
    the scoring, and counts how many it rewrote.
    """
    listed, _ = listable_options(question)
    return f"Question: {question.instructions}\nOptions:\n" + "\n".join(listed)


def unlabelled_suffix(question):
    """Ends at the colon, with no trailing space.

    A space at the end of a prefix is its own token, and BPE attaches the same
    space to the word that follows -- " Oxygen" is one token -- so the joint
    encoding of prompt and option does not begin with the prompt's own tokens and
    the option cannot be separated from it. Measured on gpt2: with the trailing
    space, none of seven ordinary options is separable; without it, all seven.
    The space belongs to the option, and `score_options` puts it there.
    """
    return unlabelled_body(question) + ANSWER_BOUNDARY


def build_content_messages(state, schema=None):
    """The system prompt for a readout that scores option content, not codes."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and options that follow. "
                "Answer with one of the options exactly as it is written. Do not "
                "explain or reason aloud."
            ),
        },
        {"role": "user", "content": state_text(state)},
    ]


# What PMI divides out. The same shape with nothing in it to answer, so what a
# score keeps is what the question added rather than what the option was worth
# before it was asked. PMI depends on this string, so it is named and recorded
# rather than inlined at the call site.
CONTENT_FREE_INSTRUCTIONS = "N/A"


def question_suffix(question, codes):
    return question_body(question, codes) + ANSWER_BOUNDARY
