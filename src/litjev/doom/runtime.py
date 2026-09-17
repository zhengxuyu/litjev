"""Everything that touches ViZDoom, so the session loop never imports the optional extra."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from litjev.doom.state import PlayerStatus, VisibleActor, visible_actors


@dataclass(frozen=True, slots=True)
class Observation:
    status: PlayerStatus
    actors: tuple[VisibleActor, ...]
    frame: str


def encode_frame(buffer) -> str:
    from PIL import Image

    array = buffer
    if array.ndim == 3 and array.shape[0] == 3:  # channel-first CRCGCB buffers
        array = array.transpose(1, 2, 0)
    image = io.BytesIO()
    Image.fromarray(array).save(image, format="JPEG", quality=82)
    return base64.b64encode(image.getvalue()).decode()


class DoomRuntime:
    """Owns the ViZDoom process and exposes only what one decision step needs."""

    def __init__(self, game, variables, max_actors: int = 6):
        self._game = game
        self._variables = variables
        self._max_actors = max_actors

    @classmethod
    def open(cls, settings) -> DoomRuntime:
        import vizdoom as vzd

        config = Path(vzd.scenarios_path) / f"{settings.scenario}.cfg"
        if not config.exists():
            raise ValueError(f"Unknown scenario '{settings.scenario}': {config} does not exist")
        game = vzd.DoomGame()
        game.load_config(str(config))
        # PLAYER mode is synchronous: the game waits for each decision, so a slow model never
        # drops frames, and a recorded episode replays at full speed whatever the latency was.
        game.set_mode(vzd.Mode.PLAYER)
        game.set_window_visible(settings.window)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(getattr(vzd.ScreenResolution, settings.resolution))
        game.set_labels_buffer_enabled(True)
        game.init()
        variables = {
            "x": vzd.GameVariable.POSITION_X,
            "y": vzd.GameVariable.POSITION_Y,
            "angle": vzd.GameVariable.ANGLE,
            "health": vzd.GameVariable.HEALTH,
            "ammo": vzd.GameVariable.AMMO2,
            "kills": vzd.GameVariable.KILLCOUNT,
        }
        return cls(game, variables, settings.max_actors)

    def buttons(self) -> list[str]:
        return [button.name for button in self._game.get_available_buttons()]

    def restart(self, recording: str | None = None) -> None:
        if recording is None:
            self._game.new_episode()
            return
        Path(recording).parent.mkdir(parents=True, exist_ok=True)
        self._game.new_episode(recording)

    def finished(self) -> bool:
        return self._game.is_episode_finished()

    def observe(self) -> Observation | None:
        state = self._game.get_state()
        if state is None:
            return None
        status = self.status()
        return Observation(
            status, visible_actors(state.labels, status), encode_frame(state.screen_buffer)
        )

    def status(self) -> PlayerStatus:
        read = self._game.get_game_variable
        return PlayerStatus(**{name: read(variable) for name, variable in self._variables.items()})

    def act(self, vector: list[int], tics: int) -> None:
        self._game.make_action(vector, tics)

    def close(self) -> None:
        self._game.close()
