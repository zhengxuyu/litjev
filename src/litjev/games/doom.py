# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Minimal Labs
# Adapted from jevlike examples/doom/environment.py at
# 94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452.
# Modified for LitJev: Gymnasium lifecycle, RGB-only observations, no training helpers.
"""Seven-button ViZDoom adapter with deterministic seeds and bounded episodes."""

from pathlib import Path
from uuid import uuid4

import vizdoom as vzd

from litjev.games.base import PixelEnv

TICS_PER_ACTION = 4
TICS_PER_SECOND = 35
DOOM_ACTIONS = (
    "turn left",
    "turn right",
    "move forward",
    "move backward",
    "strafe left",
    "strafe right",
    "attack",
)
ACTION_BUTTONS = (
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.ATTACK,
)
RESOLUTIONS = {
    "160x120": ((120, 160, 3), vzd.ScreenResolution.RES_160X120),
    "640x480": ((480, 640, 3), vzd.ScreenResolution.RES_640X480),
}


class DoomButtonsEnv(PixelEnv):
    instructions = (
        "Control Doom from the screenshot using one button. Survive, aim at visible enemies "
        "and attack them; navigate the corridor. Choose only the next controller button."
        " Attack fires straight along the crosshair, not at enemies off to the sides."
        " Turn to align an enemy with the crosshair before firing."
    )

    def __init__(
        self,
        max_steps=300,
        render_mode="rgb_array",
        scenario="deadly_corridor",
        resolution="640x480",
        recording_dir=None,
        observation_mode="rgb",
    ):
        if observation_mode not in {"rgb", "engine_text"}:
            raise ValueError("Unsupported Doom observation mode")
        self.observation_mode = observation_mode
        if observation_mode == "engine_text":
            self.instructions = self.instructions.replace(
                "the screenshot", "the engine-assisted visible-scene description"
            )
        if scenario not in {"deadly_corridor", "defend_the_center"}:
            raise ValueError("Unsupported Doom scenario")
        if resolution not in RESOLUTIONS:
            raise ValueError("Unsupported Doom resolution")
        shape, self.resolution = RESOLUTIONS[resolution]
        super().__init__(DOOM_ACTIONS, shape, max_steps, render_mode)
        self.scenario = scenario
        self.game = None
        # Separate sessions never overwrite one another's native demos.
        self.recording_dir = Path(recording_dir) / uuid4().hex if recording_dir else None
        self.recording_episode = 0
        self.recording_path = None

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("No reset options supported")
        self.close()
        super().reset(seed=seed)
        game = vzd.DoomGame()
        self.game = game
        try:
            game.load_config(str(Path(vzd.scenarios_path) / f"{self.scenario}.cfg"))
            game.set_available_buttons(list(ACTION_BUTTONS))
            game.set_window_visible(False)
            game.set_screen_format(vzd.ScreenFormat.RGB24)
            game.set_screen_resolution(self.resolution)
            game.set_labels_buffer_enabled(self.observation_mode == "engine_text")
            game.set_depth_buffer_enabled(self.observation_mode == "engine_text")
            game.set_seed(int(self.np_random.integers(0, 2**31 - 1)))
            game.init()
            if self.recording_dir is None:
                game.new_episode()
            else:
                self.recording_dir.mkdir(parents=True, exist_ok=True)
                self.recording_episode += 1
                self.recording_path = (
                    self.recording_dir / f"episode-{self.recording_episode:04d}.lmp"
                )
                # ViZDoom's native command path can hang on long demo filenames.
                # A short relative --record directory avoids this engine limitation.
                if len(str(self.recording_path).encode()) > 120:
                    raise ValueError(
                        "Native demo path is too long; use a short relative recording directory"
                    )
                game.new_episode(str(self.recording_path))
            self.frame = game.get_state().screen_buffer.copy()
        except Exception:
            self.close()
            raise
        return self.frame.copy(), {"scenario": self.scenario}

    def describe_observation(self):
        from litjev.games.doom_text import describe_buffers

        if self.observation_mode != "engine_text" or self.ended:
            raise RuntimeError("A live engine_text episode is required")
        state = self.game.get_state()
        if state is None:
            raise RuntimeError("No live Doom state to describe")
        return describe_buffers(state.labels, state.labels_buffer, state.depth_buffer)

    def step(self, action):
        self._check_action(action)
        vector = [index == int(action) for index in range(len(DOOM_ACTIONS))]
        reward = self.game.make_action(vector, TICS_PER_ACTION)
        finished = self.game.is_episode_finished()
        timeout = self.game.get_episode_timeout()
        timed_out = finished and timeout > 0 and self.game.get_episode_time() >= timeout
        terminated = finished and (self.game.is_player_dead() or not timed_out)
        outcome = (
            "death" if self.game.is_player_dead() else
            "timeout" if timed_out else
            "success" if finished else
            "step_limit" if self.steps + 1 >= self.max_steps else "running"
        )
        state = self.game.get_state()
        if state is not None:
            self.frame = state.screen_buffer.copy()
        else:
            # ViZDoom has no post-terminal screen. Explicitly retain the last real frame.
            self.frame = self.frame.copy()
        return self._transition(
            reward,
            terminated,
            {
                "game_tics": self.game.get_episode_time(),
                "terminal_frame_unavailable": state is None,
                "outcome": outcome,
            },
            timed_out=timed_out,
        )

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
        super().close()
