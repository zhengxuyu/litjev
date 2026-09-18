"""Run all default tasks with the same protocol and durable recordings."""

import argparse
import hashlib
import json
import os
import re
import shlex
import time
from collections import Counter
from pathlib import Path

import numpy as np
import vizdoom as vzd

from .perception import Perception
from .recording import Recorder
from .specs import TASK_BY_NAME, TASKS, outcome

VARIABLE_NAMES = (
    "HEALTH",
    "ARMOR",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "KILLCOUNT",
    "HITCOUNT",
    "DAMAGECOUNT",
    "POSITION_X",
    "POSITION_Y",
    "ANGLE",
)
ACTION_DESCRIPTIONS = {
    "MOVE_LEFT": "Strafe left without rotating; use to align when turning is unavailable or dodge.",
    "MOVE_RIGHT": "Strafe right without rotating; use to align when turning is unavailable or dodge.",
    "TURN_LEFT": "Rotate view left briefly (3 tics then release 1 tic).",
    "TURN_RIGHT": "Rotate view right briefly (3 tics then release 1 tic).",
    "MOVE_FORWARD": "Walk in your facing direction; avoid walls.",
    "MOVE_BACKWARD": "Walk backward without changing direction.",
    "ATTACK": "Fire the current weapon toward the crosshair; off-center enemies will be missed.",
    "NOOP": "Release all buttons and wait for 4 game tics.",
}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def load_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    for line in (Path.home() / ".zshrc").read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)$", line)
        if match:
            parts = shlex.split(match.group(1), comments=True)
            if len(parts) == 1 and "$" not in parts[0] and "`" not in parts[0]:
                return parts[0]
    raise RuntimeError("No literal TYPESAFE_API_KEY found")


class JevBackend:
    def __init__(self):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        self.client = TypeSafeClient(
            api_key=load_key(), model="jev-1.13.0", timeout=30, retry=RetryPolicy(max_retries=2)
        )

    def decide(self, state, questions):
        response = self.client.system_one(state, questions)
        return response.raw_http_response.json()


class RandomBackend:
    def __init__(self):
        self.random = np.random.default_rng(0)

    def decide(self, state, questions):
        choices = list(questions["button"]["criteria"])
        return {
            "model": "random-smoke-test",
            "usage": {},
            "answers": {
                "button": {
                    "choice": str(self.random.choice(choices)),
                    "confidence": None,
                    "probabilities": {choice: 1 / len(choices) for choice in choices},
                }
            },
        }


class QwenBackend:
    def __init__(self, url):
        import httpx

        self.client = httpx.Client(timeout=120)
        self.url = url.rstrip("/") + "/v1/systemone"

    def decide(self, state, questions):
        response = self.client.post(
            self.url, json={"model": "litjev", "state": state, "questions": questions}
        )
        response.raise_for_status()
        return response.json()


def actions_for(game):
    buttons = game.get_available_buttons()
    result = {"NOOP": [0] * len(buttons)}
    for index, button in enumerate(buttons):
        if button.name.endswith("_DELTA") or button.name in {"SPEED", "STRAFE"}:
            continue
        vector = [0] * len(buttons)
        vector[index] = 1
        result[button.name] = vector
    # One closed choice may represent a compound button press; no inert modifier actions.
    if vzd.Button.SPEED in buttons and vzd.Button.MOVE_FORWARD in buttons:
        vector = result["MOVE_FORWARD"].copy()
        vector[buttons.index(vzd.Button.SPEED)] = 1
        result["RUN_FORWARD"] = vector
    return result


def question(task, actions):
    return {
        "button": {
            "type": "choice",
            "instructions": {
                "task": task.goal,
                "control": "Choose one next action. Each decision advances 4 game tics; API waits pause "
                "the game. Turn actions rotate; strafe actions move sideways without rotation.",
                "observation": "Only visible objects are listed. Horizontal alignment does not guarantee "
                "a clear shot. Depth is 0..255, larger means more room. "
                "Audio is a stereo mixture, not a known enemy location. "
                "Use recent actions and position visits to avoid getting stuck.",
            },
            "criteria": {
                name: ACTION_DESCRIPTIONS.get(name, name.replace("_", " ").lower())
                for name in actions
            },
        }
    }


def variables(game):
    return {
        name: float(game.get_game_variable(getattr(vzd.GameVariable, name)))
        for name in VARIABLE_NAMES
    }


def latency_summary(values):
    if not values:
        return None
    return {
        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
        "mean": float(np.mean(values)),
        "first": values[0],
        "max": max(values),
    }


def run_episode(task, seed, backend, directory, smoke_steps=None):
    directory.mkdir(parents=True, exist_ok=False)
    game = vzd.DoomGame()
    recorder = None
    started = time.perf_counter()
    log = (directory / "decisions.jsonl").open("w", buffering=1)
    steps, rewards, latencies, request_latencies = [], [], [], []
    result = {"task": task.name, "seed": seed, "outcome": "infrastructure_error"}
    try:
        config = Path(vzd.scenarios_path) / f"{task.name}.cfg"
        game.load_config(str(config))
        game.set_window_visible(False)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_labels_buffer_enabled(True)
        game.set_depth_buffer_enabled(True)
        game.set_audio_buffer_enabled(True)
        game.set_audio_buffer_size(1)
        game.set_audio_sampling_rate(vzd.SamplingRate.SR_44100)
        game.set_notifications_buffer_enabled(True)
        game.set_notifications_buffer_size(64)
        if game.get_episode_timeout() == 0:
            game.set_episode_timeout(2100)
        game.set_available_game_variables([getattr(vzd.GameVariable, n) for n in VARIABLE_NAMES])
        engine_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
        game.set_seed(engine_seed)
        game.init()
        # Native recording path is relative and short to avoid ViZDoom long-path hangs.
        game.new_episode(str(directory / "episode.lmp"))
        actions = actions_for(game)
        schema = question(task, actions)
        result.update(
            skill=game.get_doom_skill(),
            timeout_tics=game.get_episode_timeout(),
            engine_seed=engine_seed,
            vizdoom_version=vzd.__version__,
        )
        save(
            directory / "manifest.json",
            {
                **result,
                "questions": schema,
                "actions": actions,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "wad_sha256": hashlib.sha256(config.with_suffix(".wad").read_bytes()).hexdigest(),
                "audio_perception": "stereo PCM energy only; no hidden monster coordinates",
                "decision_tics": 4,
                "recording_fps": 35,
                "training": False,
            },
        )
        recorder = Recorder(directory)
        perception = Perception()
        state = game.get_state()
        audio = state.audio_buffer.copy()
        notifications = state.notifications_buffer.splitlines()
        initial_tic = game.get_episode_time()
        observed_tic = initial_tic
        game_steps = 0
        decision_started = time.perf_counter()
        last_variables = variables(game)
        while state is not None:
            t0 = time.perf_counter()
            observed = perception.build(task, state, variables(game), audio, notifications)
            perception_ms = (time.perf_counter() - t0) * 1000
            frame_start = recorder.frames
            request_started = time.perf_counter()
            try:
                response = backend.decide(observed, schema)
                elapsed = (time.perf_counter() - request_started) * 1000
                request_latencies.append(elapsed)
                answer = response["answers"]["button"]
                choice = answer["choice"]
                if choice not in actions:
                    raise ValueError("Backend returned an unknown action")
            except Exception as exc:  # noqa: BLE001 - log backend failures without secret headers
                log.write(
                    json.dumps(
                        {
                            "step": len(steps),
                            "state": observed,
                            "status": "request_error",
                            "error_type": type(exc).__name__,
                            "wall_ms": (time.perf_counter() - request_started) * 1000,
                        }
                    )
                    + "\n"
                )
                result.update(outcome="api_error", error_type=type(exc).__name__)
                break
            latencies.append(elapsed)
            perception.history.append(choice)
            step_reward = 0
            chunks, messages = [], []
            for tic in range(4):
                current = game.get_state()
                if current is None:
                    break
                last_frame = current.screen_buffer.copy()
                vector = (
                    actions["NOOP"] if choice.startswith("TURN_") and tic == 3 else actions[choice]
                )
                game.set_action(vector)
                game.advance_action(1)
                game_steps += 1
                reward = float(game.get_last_reward())
                rewards.append(reward)
                step_reward += reward
                new_state = game.get_state()
                last_variables = variables(game)
                observed_tic = max(observed_tic, game.get_episode_time())
                terminal = new_state is None
                frame = last_frame if terminal else new_state.screen_buffer
                samples = (
                    np.zeros((1260, 2), dtype=np.int16) if terminal else new_state.audio_buffer
                )
                recorder.append(frame, samples, game_steps + initial_tic, len(steps), terminal)
                chunks.append(samples.copy())
                if not terminal:
                    messages.extend(new_state.notifications_buffer.splitlines())
                if game.is_episode_finished():
                    break
            row = {
                "step": len(steps),
                "state": observed,
                "response": response,
                "action": choice,
                "latency_ms": elapsed,
                "perception_ms": perception_ms,
                "request_wall_start_from_episode_seconds": request_started - decision_started,
                "video_frame_start": frame_start,
                "video_frame_end": recorder.frames,
                "reward": step_reward,
                "total_reward": sum(rewards),
                "post_action_variables": last_variables,
                "game_tics_elapsed": game_steps,
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            steps.append(row)
            if len(steps) == 1 or len(steps) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "task": task.name,
                            "seed": seed,
                            "step": len(steps),
                            "action": choice,
                            "latency_ms": round(elapsed, 1),
                        }
                    ),
                    flush=True,
                )
            state = game.get_state()
            audio = np.concatenate(chunks) if chunks else audio
            notifications = messages
            if game.is_episode_finished() or (smoke_steps and len(steps) >= smoke_steps):
                timed_out = observed_tic >= game.get_episode_timeout()
                result["outcome"] = outcome(
                    task.name,
                    dead=game.is_player_dead(),
                    timed_out=timed_out,
                    finished=game.is_episode_finished(),
                    kills=last_variables["KILLCOUNT"],
                    peak_reward=max(rewards, default=0),
                )
                break
        result.update(
            steps=len(steps),
            game_seconds=game_steps / 35,
            reward=sum(rewards),
            kills=last_variables["KILLCOUNT"],
            final_variables=last_variables,
            latency_ms=latency_summary(latencies),
            rollout_wall_seconds=time.perf_counter() - decision_started,
            actions=dict(Counter(r["action"] for r in steps)),
            model=steps[-1]["response"].get("model") if steps else None,
            usage={
                k: sum((r["response"].get("usage", {}).get(k) or 0) for r in steps)
                for k in ("input_tokens", "output_tokens")
            },
            successful_api_calls=len(steps),
            api_failures=int(result["outcome"] == "api_error"),
        )
    except Exception as exc:  # noqa: BLE001 - durable per-episode failure record
        result.update(outcome="infrastructure_error", error_type=type(exc).__name__)
        # Infrastructure messages have no API headers or credentials; keep enough to debug.
        result["error_detail"] = str(exc)[:400]
    finally:
        game.close()
        log.close()
        if recorder is not None:
            try:
                result["recording"] = recorder.close()
            except Exception as exc:  # noqa: BLE001 - preserve metrics when video export fails
                result["recording_error"] = type(exc).__name__
        result["wall_seconds_including_setup_and_export"] = time.perf_counter() - started
        save(directory / "metrics.json", result)
    print(json.dumps(result), flush=True)
    return result


def summarize(root):
    episodes = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/metrics.json"))]
    tasks = {}
    for task in TASKS:
        rows = [r for r in episodes if r["task"] == task.name]
        valid = [r for r in rows if r["outcome"] not in {"api_error", "infrastructure_error"}]
        tasks[task.name] = {
            "completed": len(valid),
            "attempted": len(rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "success_rate": (
                sum(r["outcome"] == "success" for r in valid) / len(valid)
                if valid and not task.survival
                else None
            ),
            "survived_horizon_rate": (
                sum(r["outcome"] == "survived_horizon" for r in valid) / len(valid)
                if valid and task.survival
                else None
            ),
            **{
                f"mean_{key}": float(np.mean([r[key] for r in valid])) if valid else None
                for key in ("game_seconds", "reward", "kills", "steps", "rollout_wall_seconds")
            },
        }
    report = {
        "episodes_planned": 36,
        "episodes_attempted": len(episodes),
        "tasks": tasks,
        "episodes": episodes,
        "billing_cost": None,
        "notes": [
            "Survival tasks have no goal-completion success rate.",
            "SDK retries and network latency are included in decision latency.",
            "Do not compare task rewards across different reward functions.",
            "Three runs per task are exploratory, not a precise success estimate.",
            "Video clock is game time; request wall times are recorded separately.",
            "Terminal frame repeats last real image; terminal audio is zero-padded.",
        ],
    }
    save(root / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jev", "qwen", "random"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18800")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASK_BY_NAME), default=list(TASK_BY_NAME)
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # All assets stay under this working directory; demo paths must remain short.
    os.chdir(args.output)
    backend = {"jev": JevBackend, "random": RandomBackend, "qwen": lambda: QwenBackend(args.url)}[
        args.backend
    ]()
    consecutive_errors = 0
    for name in args.tasks:
        for seed in range(args.episodes):
            directory = Path(name) / f"seed-{seed}"
            if (directory / "metrics.json").exists():
                continue
            if directory.exists():
                raise RuntimeError(
                    f"Incomplete episode exists: {directory}; preserve it before retry"
                )
            result = run_episode(TASK_BY_NAME[name], seed, backend, directory, args.smoke_steps)
            summarize(Path("."))
            consecutive_errors = (
                consecutive_errors + 1
                if result["outcome"] in {"api_error", "infrastructure_error"}
                else 0
            )
            if consecutive_errors >= 2:
                raise RuntimeError(
                    "Stopped after two infrastructure/API failures; recordings preserved"
                )
    print("SUITE_FINISHED", flush=True)


if __name__ == "__main__":
    main()
