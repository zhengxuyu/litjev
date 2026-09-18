# Recorded twelve-task ViZDoom evaluation

This is an **engine-assisted text-observation** benchmark, not a raw-image model benchmark.
Both Jev and Qwen/LitJev receive the same task-specific Choice schema and perception code.
Run the twelve default single-player tasks with seeds 0, 1, 2 (36 episodes per backend).

```sh
uv run --script scripts/run_doom_suite.py --backend jev --output /absolute/results/jev
uv run --script scripts/run_doom_suite.py --backend qwen --url http://127.0.0.1:18800 --output /absolute/results/qwen
```

The standalone script has a uv lockfile and avoids installing Torch for hosted-API evaluation.
Requires `ffmpeg` on PATH. Export `TYPESAFE_API_KEY`; a literal assignment in `~/.zshrc` is also
supported without sourcing that file. Credentials are never written to traces.
On macOS, audio buffers require OpenAL Soft; for Homebrew installations use
`DYLD_LIBRARY_PATH=/opt/homebrew/opt/openal-soft/lib`. Do not silently evaluate Basic Audio with
a broken audio backend. The SDK currently uses pinned `jev-1.13.0`, 30-second timeout and at most
two retries. End-to-end latency includes SDK retries.

## Protocol

- Keep each installed scenario's default difficulty, WAD, rewards, and button list.
- Add NOOP; represent non-delta buttons as discrete choices. Omit standalone inert SPEED/STRAFE
  modifiers and continuous delta controls; add RUN_FORWARD where supported. The manifest records
  exact button vectors. This is a custom closed-choice action protocol, not the default Gym action space.
- Decisions advance four tics; turn buttons release for the fourth tic. API waits pause simulation.
- Preserve configured timeouts; set 2100 tics (60 seconds) for scenarios with no timeout.
- Render at 640×480; capture every simulation tic at 35 FPS, independently of API latency.
- Perception uses visible label masks, object metadata, depth, player variables, recent actions,
  and coarse visited-position counts. No scenario-specific hidden enemy coordinates or goal oracle.
- Basic Audio explicitly removes monster labels and uses stereo PCM energy. This is a simple
  engineered audio representation, not equivalent to a learned audio model. Basic Notifications uses
  the actual `Shoot:` notification. No target notification means the instructions tell the model to wait.
- Goal tasks: success, failed objective, death, timeout. Survival tasks: kills, reward, survival duration
  and horizon survival; **do not assign a goal-completion rate** to these tasks.
- Record infrastructure/API errors separately, preserving partial artifacts; stop on two consecutive
  such failures. Never execute a random fallback as if it were a model decision.

## Artifacts

Each `<task>/seed-N/` directory contains:

| File | Contents |
|---|---|
| `replay.mp4` | H.264 35 FPS video with audio, ready for editing |
| `video.mp4`, `audio.wav` | Separate video and lossless PCM sources |
| `episode.lmp` | ViZDoom native replay, finalized when the environment closes |
| `frames.jsonl` | Every frame's video time, game tic and corresponding decision |
| `decisions.jsonl` | Exact text state, raw API JSON, action, reward, token usage, latency and video-frame interval |
| `manifest.json` | Task schema, button vectors, seed, difficulty, timeout and asset hashes |
| `metrics.json` | Outcome, reward, kills, steps, game/wall times and latency mean/P50/P95/P99 |

`summary.json` is refreshed after each episode. MP4 files are finalized at the end of a run;
don't expect a currently recording MP4 to be playable. The engine has no final terminal observation:
the last image is repeated and terminal audio is zero-padded for that tic, explicitly marked in
`frames.jsonl`. Frames represent game time; use decision timestamps to visualize real API waiting.

Three trials per task are exploratory. Compare matching tasks and seeds, not raw rewards across
tasks. Network-inclusive hosted latency is not pure model compute time. Billing, unreported retries,
and hosted raw logits are unavailable and must not be fabricated. Qwen's earlier corridor-only
skill-3 results are **not** part of this default-difficulty twelve-task comparison.

## Smoke validation

```sh
uv run --script scripts/run_doom_suite.py --backend random --output /absolute/smoke --episodes 1 --smoke-steps 8
```

Smoke runs are explicitly labeled random and must not enter model results. Resume skips existing
per-episode metrics; an unfinished directory requires inspection and preservation before retry.
