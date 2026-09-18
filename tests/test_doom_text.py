from types import SimpleNamespace

import numpy as np
import pytest

from litjev.games.doom_text import describe_buffers


def test_description_uses_visible_masks_not_world_coordinates():
    labels = np.zeros((12, 18), dtype=np.uint8)
    labels[3:6, 2:5] = 7
    depth = np.full_like(labels, 100)
    depth[labels == 7] = 30
    visible = SimpleNamespace(value=7, object_name="Zombie", object_position_x=999999)
    hidden = SimpleNamespace(value=8, object_name="HiddenMonster")
    text = describe_buffers([visible, hidden], labels, depth)
    assert "Zombie" in text and "left" in text and "30" in text
    assert "HiddenMonster" not in text and "999999" not in text
    assert "raw" in text and "meters" in text
    with pytest.raises(ValueError):
        describe_buffers([], None, depth)


def test_real_doom_text_trace_never_passes_pixels_to_client():
    pytest.importorskip("vizdoom")
    from test_games import RecordingClient

    from litjev.games import make_env
    from litjev.games.policy import LitJevPolicy
    from litjev.games.rollout import record_episode

    client = RecordingClient()
    original = client.decide
    inputs = []

    def decide(state, schema, image):
        inputs.append((state, image))
        return original(state, schema, np.zeros((1, 1, 3), np.uint8))

    client.decide = decide
    with make_env("doom", max_steps=2, observation_mode="engine_text") as env:
        policy = LitJevPolicy(client, env.unwrapped.action_names, env.unwrapped.instructions)
        trace = record_episode(env, policy, seed=0)
    assert trace["observation"] == "engine_labels_depth_text"
    assert len(inputs) == 2 and all(image is None for _, image in inputs)
    assert all("observation" in state for state, _ in inputs)
    assert all(row["observation_text"] for row in trace["decisions"])
    assert all(row["description_ms"] >= 0 for row in trace["decisions"])
    assert trace["decisions"][-1]["info"]["outcome"] == "step_limit"


def test_local_text_client_does_not_construct_visual_state():
    from dataclasses import dataclass

    from litjev.games.policy import LocalDecisionClient

    @dataclass
    class Result:
        ok: bool = True

    class Engine:
        model_id = "test"

        def evaluate(self, state, schema):
            assert state == "scene"
            return Result()

    result = LocalDecisionClient(Engine()).decide(
        "scene", {"a": {"type": "choice", "criteria": {"left": None}}}, None
    )
    assert result == {"ok": True}


def test_doom_instructions_explain_attack_mechanics():
    from litjev.games.doom import DoomButtonsEnv

    assert "crosshair" in DoomButtonsEnv.instructions
    assert "before firing" in DoomButtonsEnv.instructions
