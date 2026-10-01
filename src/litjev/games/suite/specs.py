"""Shared task protocol: independent of the decision backend."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Task:
    name: str
    goal: str
    survival: bool = False


TASKS = (
    Task("basic", "Strafe to align with the visible monster, then shoot to kill it."),
    Task(
        "basic_audio",
        "Locate the invisible monster using stereo sound. Strafe to align, "
        "then shoot. No visual monster locations are provided; compare sound over time.",
    ),
    Task(
        "basic_notifications",
        "Read the on-screen notification and kill ONLY the specified "
        "monster, not the other monsters. If no target notification has arrived, wait. Strafe to aim.",
    ),
    Task(
        "deadly_corridor",
        "Reach the green armor at the far end of the straight corridor "
        "running along world +X. Survive the enemies on both sides.",
    ),
    Task(
        "deathmatch",
        "Kill as many monsters as possible before death or the time limit. "
        "Collect useful health, ammunition and weapons.",
        True,
    ),
    Task(
        "defend_the_center",
        "Turn and shoot approaching enemies. Maximize kills and survival; ammunition is limited.",
        True,
    ),
    Task(
        "defend_the_line",
        "Turn and shoot approaching enemies. Maximize kills and survival; ammunition is limited.",
        True,
    ),
    Task("health_gathering", "Stay alive on a damaging floor by collecting health packs.", True),
    Task(
        "health_gathering_supreme",
        "Stay alive on a damaging floor; navigate the complex layout to collect health packs.",
        True,
    ),
    Task(
        "my_way_home",
        "Explore connected rooms and find the green armor. Avoid repeatedly "
        "visiting the same locations. The goal is not necessarily along world +X.",
    ),
    Task(
        "predict_position",
        "You have one rocket. Predict the moving monster position and "
        "lead the target so the rocket hits. Wait or turn before firing if necessary.",
    ),
    Task("take_cover", "Dodge incoming projectiles by strafing. Maximize survival time.", True),
)
TASK_BY_NAME = {task.name: task for task in TASKS}


def outcome(name, *, dead, timed_out, finished, kills, peak_reward):
    if dead:
        return "death"
    if TASK_BY_NAME[name].survival:
        return "survived_horizon" if timed_out else "ended" if finished else "step_limit"
    if timed_out:
        return "timeout"
    if not finished:
        return "step_limit"
    if name == "basic_notifications":
        success = peak_reward > 50
    elif name in {"basic", "basic_audio", "predict_position"}:
        success = kills > 0
    elif name == "my_way_home":
        success = peak_reward > 0.5
    else:
        success = peak_reward > 900
    return "success" if success else "failed_objective"


def audio_description(samples):
    """Only observed stereo PCM energy; never hidden actor positions."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("Expected samples x stereo channels")
    rms = np.sqrt(np.mean(samples * samples, axis=0))
    balance = float((rms[1] - rms[0]) / max(float(rms.sum()), 1))
    direction = (
        "silent"
        if max(rms) < 1
        else "balanced"
        if abs(balance) < 0.08
        else "right"
        if balance > 0
        else "left"
    )
    return {
        "left_rms": round(float(rms[0]), 2),
        "right_rms": round(float(rms[1]), 2),
        "balance_right_minus_left": round(balance, 3),
        "louder_channel": direction,
        "note": "Mixture of all game sounds, including your own gun; not a target oracle.",
    }
