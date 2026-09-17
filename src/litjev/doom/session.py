"""One background episode loop: read the state, ask for one action, step the game."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from litjev.doom.actions import ActionMenu
from litjev.doom.runtime import DoomRuntime
from litjev.doom.state import render_state
from litjev.schema import DecisionSchema

FIELD = "action"
BRIEFING = (
    "You are a marine holding the centre of a circular arena. Monsters close in from every "
    "side. Turn to face the nearest one and shoot it before it reaches you, and do not waste "
    "ammunition on empty air."
)


@dataclass(frozen=True, slots=True)
class DoomSettings:
    scenario: str = "defend_the_center"
    briefing: str = BRIEFING
    tics: int = 4
    resolution: str = "RES_640X480"
    max_actors: int = 6
    window: bool = False
    recording: str | None = None

    def episode_path(self, episode: int) -> str | None:
        if self.recording is None:
            return None
        return str(Path(self.recording) / f"episode-{episode:03d}.lmp")


class DoomSession:
    """Runs the episode on its own thread and publishes one payload per decision."""

    def __init__(self, engine_provider, settings=None, runtime_factory=DoomRuntime.open):
        self.engine_provider = engine_provider
        self.settings = settings or DoomSettings()
        self.runtime_factory = runtime_factory
        self._condition = threading.Condition()
        self._version = 0
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._condition:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="doom-session", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 20.0) -> None:
        thread = self._thread
        self._stop.set()
        if thread is not None:
            thread.join(timeout)

    def watch(self):
        """Follow every payload published from now on; None is an idle keep-alive tick."""
        # The cursor is taken eagerly so nothing published between this call and the first
        # iteration of the returned generator is skipped.
        with self._condition:
            return self._follow(self._version)

    def _follow(self, seen: int):
        while True:
            with self._condition:
                if self._version == seen:
                    self._condition.wait(timeout=5.0)
                if self._version == seen:
                    yield None
                    continue
                seen, payload = self._version, self._latest
            yield payload
            if payload.get("event") == "stopped":
                return

    def _publish(self, payload: dict) -> None:
        with self._condition:
            self._version += 1
            self._latest = payload
            self._condition.notify_all()

    def _run(self) -> None:
        runtime = None
        failure = None
        try:
            runtime = self.runtime_factory(self.settings)
            menu = ActionMenu.compile(runtime.buttons())
            schema = DecisionSchema.from_mapping({FIELD: menu.field()})
            catalog = menu.catalog()
            engine = self.engine_provider()
            episode, step = 0, 0
            runtime.restart(self.settings.episode_path(episode))
            while not self._stop.is_set():
                if runtime.finished():
                    episode += 1
                    runtime.restart(self.settings.episode_path(episode))
                observation = runtime.observe()
                if observation is None:
                    continue
                text = render_state(
                    self.settings.briefing,
                    observation.status,
                    observation.actors,
                    self.settings.max_actors,
                )
                started = perf_counter()
                result = engine.decide(text, schema)
                elapsed = perf_counter() - started
                answer = result.answers[FIELD]
                step += 1
                self._publish(
                    {
                        "event": "step",
                        "step": step,
                        "episode": episode,
                        "frame": observation.frame,
                        "action": answer.value,
                        "move": menu.move(answer.value),
                        "menu": catalog,
                        "probabilities": answer.probabilities,
                        "gamma": answer.gamma,
                        "decision_seconds": elapsed,
                        "forward_calls": result.usage.forward_calls,
                        "output_tokens": result.usage.output_tokens,
                        "input_tokens": result.usage.input_tokens,
                        "state_text": text,
                        "status": observation.status.summary(),
                    }
                )
                runtime.act(menu.vector(answer.value), self.settings.tics)
        # A worker thread must not die silently; every failure is reported to the browser.
        except Exception as error:  # noqa: BLE001
            failure = f"{type(error).__name__}: {error}"
        finally:
            if runtime is not None:
                runtime.close()
            self._thread = None
            self._publish({"event": "stopped", "error": failure})
