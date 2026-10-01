"""The SGLang backend's logic, without an engine.

Everything that can be wrong without a GPU is wrong here: how candidate
logprobs are placed, which position and layer the hidden state comes from, and
what happens when the engine returns less than the head needs.
"""
import numpy as np
import pytest

from inference.sglang_backend import SGLangScorer, SGLangSettings


def scorer(layers=(48,), batch=False):
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


def test_the_hidden_state_comes_from_the_last_prompt_position_and_the_named_layer():
    # [position, layer, width]: two positions, the second being the generated token.
    states = np.zeros((2, 64, 4), dtype=np.float32)
    states[0, 48] = [1, 2, 3, 4]      # the readout position
    states[1, 48] = [9, 9, 9, 9]      # the generated token, which must not be read
    hidden = scorer()._hidden_from({"meta_info": {"hidden_states": states.tolist()}})
    assert hidden.shape == (1, 4)
    assert hidden[0].tolist() == [1, 2, 3, 4]


def test_a_last_layer_only_engine_refuses_a_head_that_reads_deeper():
    states = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="only the last hidden layer"):
        scorer(layers=(48,))._hidden_from({"meta_info": {"hidden_states": states.tolist()}})


def test_a_last_layer_only_engine_serves_a_head_trained_on_the_last_layer():
    states = np.array([[1.0, 2.0], [7.0, 7.0]], dtype=np.float32)
    hidden = scorer(layers=(-1,))._hidden_from({"meta_info": {"hidden_states": states.tolist()}})
    assert hidden.tolist() == [[1.0, 2.0]]


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
    """The server returns one layer; reading another's vector as it fails silently."""
    from inference import SGLangServerScorer, SGLangServerSettings

    with pytest.raises(ValueError, match="only the last hidden layer"):
        SGLangServerScorer.load(SGLangServerSettings(
            base_url="http://localhost:30000", model_id="m", feature_layers=(48,)))


def test_an_absent_server_says_how_to_start_one():
    from inference.http_backend import _HttpEngine

    engine = _HttpEngine("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(RuntimeError, match="launch_server"):
        engine.generate("hi", sampling_params={"max_new_tokens": 1})
