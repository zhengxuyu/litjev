"""The SGLang backend's logic, without an engine.

Everything that can be wrong without a GPU is wrong here: how candidate
logprobs are placed, which position and layer the hidden state comes from, and
what happens when the engine returns less than the head needs.
"""
import numpy as np
import pytest

from inference.sglang_backend import SGLangScorer, SGLangSettings


def scorer(layers=(-1,), batch=False):
    """Layer -1 by default, the only layer this backend can serve."""
    settings = SGLangSettings(model_id="m", feature_layers=layers, batch_questions=batch)
    return SGLangScorer(engine=None, tokenizer=None, settings=settings, hidden_size=5120)


# The shape the engine actually returns: the
# candidates live under output_token_ids_logprobs as (logprob, token_id, text),
# and input_token_ids_logprobs sits beside it holding [None].
def real_meta(rows):
    return {"meta_info": {"output_token_ids_logprobs": [rows],
                          "input_token_ids_logprobs": [None]}}


def test_candidate_logprobs_are_ordered_by_code_not_by_the_engine():
    output = real_meta([(-2.0, 77, None), (-0.5, 33, None)])
    logits = SGLangScorer._logits_from(output, [33, 77])
    assert logits.tolist() == pytest.approx([-0.5, -2.0])


def test_the_none_beside_the_real_row_is_not_read_instead_of_it():
    """Reading input_token_ids_logprobs[-1] is how this crashed on the first step."""
    output = real_meta([(-7.75, 351, None), (-12.25, 357, None)])
    logits = SGLangScorer._logits_from(output, [357, 351])
    assert logits.tolist() == pytest.approx([-12.25, -7.75])


def test_a_missing_candidate_logprob_is_an_error_not_a_zero():
    with pytest.raises(RuntimeError, match="omitted logprobs"):
        SGLangScorer._logits_from(real_meta([(-0.5, 33, None)]), [33, 77])


def test_absent_logprobs_say_what_was_not_passed():
    with pytest.raises(RuntimeError, match="token_ids_logprob"):
        SGLangScorer._logits_from({"meta_info": {}}, [1])
    with pytest.raises(RuntimeError, match="all None"):
        SGLangScorer._logits_from(
            {"meta_info": {"output_token_ids_logprobs": [None],
                           "input_token_ids_logprobs": [None]}}, [1])


def test_the_hidden_state_comes_from_the_last_prompt_position():
    """The measured shape, not an assumed one. It was probed by what moves with
    the prompt rather than by rank: a 15-token prompt gives [1, 15, 5120], so the
    middle axis is position and there is no depth axis. The fixture this replaces
    built [position, layer, width], which is the shape the old parser was written
    for and the reason it read position 48 as layer 48."""
    states = np.zeros((1, 15, 4), dtype=np.float32)
    states[0, -1] = [1, 2, 3, 4]      # the readout: the last prompt position
    states[0, 0] = [9, 9, 9, 9]
    hidden = scorer(layers=(-1,))._hidden_from({"meta_info": {"hidden_states": states.tolist()}})
    assert hidden.shape == (1, 4)
    assert hidden[0].tolist() == [1, 2, 3, 4]


def test_a_head_that_reads_deeper_is_refused_at_construction():
    """Under the old parser this was silent: `feature_layers=(48,)` returned the
    vector at position 48, or raised IndexError on a prompt of fewer than 49
    tokens. There is no depth axis to index, so it is refused."""
    with pytest.raises(ValueError, match="only the last hidden layer"):
        scorer(layers=(48,))


def test_a_head_that_reads_deeper_is_refused_before_the_engine_is_loaded(monkeypatch):
    """The ordering, which the construction test cannot see because it passes
    `engine=None`. `load` used to reach the guard only after `sgl.Engine(...)`, so
    the refusal arrived after a full model load. `sglang` is replaced
    by a module that fails if anything touches it."""
    import sys
    import types

    def explode(*arguments, **keywords):
        raise AssertionError("the engine was loaded before the layer was checked")

    fake = types.ModuleType("sglang")
    fake.Engine = explode
    monkeypatch.setitem(sys.modules, "sglang", fake)
    with pytest.raises(ValueError, match="only the last hidden layer"):
        SGLangScorer.load(SGLangSettings(model_id="m", feature_layers=(48,)))


def test_the_parser_refuses_a_deeper_layer_too_and_not_only_the_constructor():
    """The constructor is the guard; this is the backstop, so a caller that reaches
    the parser another way cannot get position 48 back as though it were layer 48."""
    from inference.sglang_backend import readout_hidden_state

    states = np.zeros((1, 60, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="no depth axis"):
        readout_hidden_state(states, (48,))


def test_a_second_generated_step_does_not_move_the_readout():
    """`states[0, -1]` whatever the step axis holds: the readout is the last prompt
    position of the prefill, and the old parser's `states[-2] if shape[0] > 1`
    reached into a different axis to try to say the same thing."""
    from inference.sglang_backend import readout_hidden_state

    states = np.zeros((2, 15, 4), dtype=np.float32)
    states[0, -1] = [1, 2, 3, 4]
    states[1, -1] = [8, 8, 8, 8]
    assert readout_hidden_state(states).tolist() == [1, 2, 3, 4]


def test_a_two_dimensional_return_is_read_as_positions():
    """Unseen on the measured build. It is the same reading as `verify.py`'s, which
    is the point of there being one parser."""
    from inference.sglang_backend import readout_hidden_state

    states = np.array([[1.0, 2.0], [7.0, 7.0]], dtype=np.float32)
    assert readout_hidden_state(states).tolist() == [7.0, 7.0]


def test_an_unexpected_rank_says_how_to_identify_it():
    from inference.sglang_backend import readout_hidden_state

    with pytest.raises(RuntimeError, match="prompt length"):
        readout_hidden_state(np.zeros((2, 3, 4, 5), dtype=np.float32))


def test_no_feature_layers_means_no_hidden_state_is_asked_for():
    assert scorer(layers=())._hidden_from({"meta_info": {}}) is None


def test_batching_is_off_by_default_because_the_engine_has_open_bugs_there():
    assert SGLangSettings(model_id="m").batch_questions is False


def test_the_hidden_size_is_read_from_the_checkpoint_not_from_transformers(tmp_path):
    """SGLang pins a transformers that refuses architectures the engine serves fine."""
    import json

    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "something_new", "text_config": {"hidden_size": 5120}}))
    assert SGLangScorer._hidden_size(str(tmp_path), "main") == 5120


def test_a_flat_config_works_too(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 2560}))
    assert SGLangScorer._hidden_size(str(tmp_path), "main") == 2560


def test_a_config_without_a_width_says_so(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))
    with pytest.raises(ValueError, match="no hidden_size"):
        SGLangScorer._hidden_size(str(tmp_path), "main")


def test_the_server_scorer_refuses_a_head_that_reads_deeper_than_the_last_layer():
    """The server returns one layer; reading another's vector as it fails silently.
    Refused at `load`, before the tokenizer is fetched, as well as by the shared
    constructor underneath it."""
    from inference import SGLangServerScorer, SGLangServerSettings

    with pytest.raises(ValueError, match="only the last hidden layer"):
        SGLangServerScorer.load(SGLangServerSettings(
            base_url="http://localhost:30000", model_id="m", feature_layers=(48,)))


def test_an_absent_server_says_how_to_start_one():
    from inference.http_backend import _HttpEngine

    engine = _HttpEngine("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(RuntimeError, match="launch_server"):
        engine.generate("hi", sampling_params={"max_new_tokens": 1})


# ------------------------------------------------- the content readout on SGLang
#
# The readout the engine uses by default scores an option by its own text, so
# SGLang has to implement it too or the engine refuses to build on it.
#
# None of this needs a GPU: what can be wrong without one is which tokens are
# sent, which come back, and whether the sum is over the option's own.


class CharTokenizer:
    """Characters as tokens, and a chat template that is a plain wrapper."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return "".join(message["content"] for message in messages) + "\n"


class FakeEngine:
    """Replies shaped like SGLang's, recording what it was asked.

    `whole_prompt` returns an entry for every input position rather than from
    `logprob_start_len`, which is the reading of that parameter this code must not
    depend on.
    """

    def __init__(self, logprob=-1.0, whole_prompt=False):
        self.calls = []
        self.logprob = logprob
        self.whole_prompt = whole_prompt

    def generate(self, input_ids=None, sampling_params=None, return_logprob=False,
                 logprob_start_len=None, **kwargs):
        self.calls.append({"input_ids": input_ids, "sampling_params": sampling_params,
                           "return_logprob": return_logprob,
                           "logprob_start_len": logprob_start_len})
        outputs = []
        for row, start in zip(input_ids, logprob_start_len):
            first = 0 if self.whole_prompt else start
            outputs.append({"meta_info": {
                "input_token_logprobs": [[self.logprob, token, None]
                                         for token in row[first:]]}})
        return outputs


def content_scorer(engine, batch=False):
    settings = SGLangSettings(model_id="m", feature_layers=(), batch_questions=batch)
    return SGLangScorer(engine=engine, tokenizer=CharTokenizer(), settings=settings,
                        hidden_size=8)


def a_question():
    from litjev.schema import Choice

    return Choice(instructions="What is 2+2?",
                  criteria={"A": "three", "B": "four", "C": "five point one"})


def test_only_the_options_own_tokens_are_summed():
    """The prefix is thousands of tokens and identical under every option, so
    counting it would hide the error in a constant that cancels in the ranking."""
    engine = FakeEngine(whole_prompt=True)
    scores, rewritten = content_scorer(engine)._score_options("Answer.", a_question())
    assert rewritten == 0
    assert [score.text for score in scores] == ["three", "four", "five point one"]
    # The leading space belongs to the option, so " three" is six characters.
    assert [score.tokens for score in scores] == [6, 5, 15]
    assert [score.total for score in scores] == [-6.0, -5.0, -15.0]


def test_the_tail_is_found_by_token_id_and_not_by_the_offset_asked_for():
    """`logprob_start_len` has moved between SGLang versions. The same scores come
    back whether the engine honours it or returns the whole prompt."""
    honoured = content_scorer(FakeEngine())._score_options("Answer.", a_question())[0]
    whole = content_scorer(FakeEngine(whole_prompt=True))._score_options(
        "Answer.", a_question())[0]
    assert [s.total for s in honoured] == [s.total for s in whole]


def test_one_request_per_option_while_batching_is_off():
    """An option shares a forward pass here exactly as a question does, and the
    engine's misattribution bugs are about that sharing."""
    engine = FakeEngine()
    content_scorer(engine)._score_options("Answer.", a_question())
    assert len(engine.calls) == 3
    assert all(len(call["input_ids"]) == 1 for call in engine.calls)

    together = FakeEngine()
    content_scorer(together, batch=True)._score_options("Answer.", a_question())
    assert len(together.calls) == 1
    assert len(together.calls[0]["input_ids"]) == 3


def test_the_rows_are_token_ids_and_every_one_carries_the_whole_prefix():
    """Sent as ids rather than as text: re-tokenising `prefix + option` server-side
    could merge across the boundary that was just proved separable."""
    engine = FakeEngine()
    content_scorer(engine)._score_options("Answer.", a_question())
    rows = [call["input_ids"][0] for call in engine.calls]
    prefix_length = len(rows[0]) - 6
    prefix = rows[0][:prefix_length]
    assert all(row[:prefix_length] == prefix for row in rows)
    assert all(call["logprob_start_len"] == [prefix_length - 1] for call in engine.calls)
    assert all(call["return_logprob"] for call in engine.calls)
    assert all(call["sampling_params"][0]["temperature"] == 0.0 for call in engine.calls)


def test_the_prompt_shows_the_options_with_no_label_against_them():
    """The letter is what the coded readout's position preference attaches to, so
    a content row that still carried one would be measuring the other readout."""
    engine = FakeEngine()
    content_scorer(engine)._score_options("Answer.", a_question())
    prompt = "".join(chr(token) for token in engine.calls[0]["input_ids"][0])
    assert "Options:\nthree\nfour\nfive point one" in prompt
    assert "\nA three" not in prompt and '"A"' not in prompt
    assert prompt.endswith(" three"), "the row is the prefix and then the option itself"


def test_the_content_free_pass_is_the_same_options_after_an_empty_question():
    engine = FakeEngine()
    scores, _ = content_scorer(engine).score_options("Answer.", a_question())
    assert len(engine.calls) == 6, "three options, twice"
    assert all(score.prior is not None for score in scores)
    blank = "".join(chr(token) for token in engine.calls[3]["input_ids"][0])
    assert "N/A" in blank, "the content-free instructions, named in prompting.py"


def test_the_content_free_pass_can_be_turned_off():
    engine = FakeEngine()
    scores, _ = content_scorer(engine).score_options("Answer.", a_question(),
                                                     content_free=False)
    assert len(engine.calls) == 3
    assert all(score.prior is None for score in scores)


def test_a_reply_about_other_tokens_is_refused_rather_than_summed():
    """A plausible number from the wrong positions is the failure this port could
    have, and it would not show up anywhere downstream."""
    output = {"meta_info": {"input_token_logprobs": [[-1.0, 11, None], [-1.0, 12, None]]}}
    with pytest.raises(RuntimeError, match="not aligned"):
        SGLangScorer._option_total(output, [11, 13])


def test_a_reply_that_begins_after_the_option_is_refused():
    output = {"meta_info": {"input_token_logprobs": [[-1.0, 13, None]]}}
    with pytest.raises(RuntimeError, match="begins after the option"):
        SGLangScorer._option_total(output, [11, 13])


def test_a_none_among_the_option_tokens_is_refused():
    output = {"meta_info": {"input_token_logprobs": [[None, 11, None], [-1.0, 13, None]]}}
    with pytest.raises(RuntimeError, match="None for one of the option"):
        SGLangScorer._option_total(output, [11, 13])


def test_no_input_logprobs_says_what_was_not_passed():
    with pytest.raises(RuntimeError, match="return_logprob"):
        SGLangScorer._option_total({"meta_info": {}}, [11])


def test_the_engine_now_builds_on_sglang_under_the_default_readout():
    """The guard at engine construction refuses a scorer without `score_options`."""
    from litjev.decision import SchemaDecisionEngine

    engine = SchemaDecisionEngine(content_scorer(FakeEngine()))
    assert engine.readout == "content"


def test_the_server_sends_ids_and_the_start_position_over_http(monkeypatch):
    """The server scorer inherits the readout, so its transport has to carry both."""
    import io
    import json as json_module
    import urllib.request

    from inference.http_backend import _HttpEngine

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured.update(json_module.loads(request.data))
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda request, timeout=None: _closing(fake_urlopen(request, timeout)))
    _HttpEngine("http://localhost:1", timeout=1.0).generate(
        input_ids=[[1, 2, 3]], sampling_params=[{"max_new_tokens": 1}],
        return_logprob=True, logprob_start_len=[1])
    assert captured["input_ids"] == [[1, 2, 3]]
    assert captured["logprob_start_len"] == [1]
    assert "text" not in captured, "a row of ids must not be re-tokenised from text"


class _closing:
    """`urlopen` is used as a context manager, which BytesIO is not."""

    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *exception):
        return False


# ------------------------------------------------- what the verification requires
#
# These are the rules `inference.verify` applies to the content readout, which are about the decision rather
# than about a float: the two backends differ by a few percent on one activation
# and a summed score inherits that, so requiring the sums to match would be
# requiring something untrue.


def test_the_verification_passes_when_both_backends_choose_the_same_option():
    from inference.verify import report_options

    reference = [[(0, 2, -4.0), (1, 5, -9.0)]]
    candidate = [[(0, 2, -4.2), (1, 5, -9.3)]]
    ok, summary = report_options(reference, candidate)
    assert ok and summary["same_choice"]
    assert summary["sum_difference_max"] == pytest.approx(0.3)


def test_the_verification_fails_when_the_chosen_option_moves():
    """A sum that drifts is the implementations differing; a choice that moves is
    the readout answering a different question."""
    from inference.verify import report_options

    ok, summary = report_options([[(0, 2, -4.0), (1, 5, -4.1)]],
                                 [[(0, 2, -4.2), (1, 5, -4.0)]])
    assert not ok and not summary["same_choice"]


def test_a_different_token_count_fails_before_the_scores_are_compared():
    """The same tokenizer produced both sides, so a different count is a different
    prompt, and comparing the sums would be comparing two different questions."""
    from inference.verify import report_options

    ok, _ = report_options([[(0, 2, -4.0)]], [[(0, 3, -4.0)]])
    assert not ok


def test_the_verification_questions_have_options_of_uneven_length():
    """An off-by-one in the offset costs every option one token, which equal
    lengths would hide in a constant that cancels in the ranking."""
    from inference.verify import option_questions

    for _, question in option_questions():
        lengths = {len(text) for text in question.descriptions}
        assert len(lengths) > 1


def test_verify_and_the_backend_read_the_shape_the_same_way():
    """The defect this pair replaces was two parsers disagreeing: this module read
    a 3-D return as [position, layer, width] and verify.py as [step, position,
    width]. They cannot diverge again without this failing."""
    import inspect

    from inference import verify
    from inference.sglang_backend import readout_hidden_state

    assert "readout_hidden_state" in inspect.getsource(verify.sglang_hidden)
    states = np.zeros((1, 15, 4), dtype=np.float32)
    states[0, -1] = [5, 6, 7, 8]
    assert readout_hidden_state(states).tolist() == [5, 6, 7, 8]


def test_verify_refuses_a_layer_this_engine_cannot_select():
    """It used to accept --layer 48 and ignore it, then report MISALIGNED because
    the reference was layer 48 and the candidate was the last layer."""
    from inference.verify import sglang_hidden

    with pytest.raises(SystemExit, match="only"):
        sglang_hidden(engine=None, prompts=["x"], layer=48, batched=False)


def test_the_option_margin_is_reported_beside_the_choice():
    """A flip between two options a hair apart is a tie resolved differently; a flip
    on a wide margin is a fault. The line that reported only the choice could not
    say which had happened."""
    from inference.verify import report_options

    ok, summary = report_options([[(0, 2, -4.0), (1, 5, -4.0001)]],
                                 [[(0, 2, -4.0001), (1, 5, -4.0)]])
    assert not ok
    (row,) = summary["per_question"]
    assert row["margin_reference"] < 0.001 and row["margin_candidate"] < 0.001
    assert (row["chosen_reference"], row["chosen_candidate"]) == (0, 1)
