# Visual Gym environments

These examples port the Doom, chess-controller, and film workflow from
[jevlike](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/examples).
See [third-party notices](../THIRD_PARTY_NOTICES.md) for source provenance and MIT
attribution. No upstream trained weights, training scripts, or soundtrack are used.
The policy uses your off-the-shelf Qwen checkpoint without fine-tuning.

## Install and run

From the repository root, with Python 3.11+, uv and sufficient GPU memory:

```bash
uv sync --locked --extra games
# Terminal 1: Qwen service, API, screenshot playground and replay viewer.
uv run --locked --extra games litjev --model Qwen/Qwen3.8-27B
# Terminal 2: games call that service one decision at a time.
uv run --locked --extra games litjev-play chess --max-steps 100 --output runs/chess.json
uv run --locked --extra games litjev-play doom --max-steps 100 --output runs/doom.json
```

An existing checkpoint is accepted with `--model /path/to/checkpoint`. The server
needs the checkpoint's vision processor files as well as its model weights.
For a remote server, forward its port with SSH and pass `--url http://127.0.0.1:PORT`.
No server is needed for local Python inference:

```bash
uv run --locked --extra games litjev-play chess --model /path/to/checkpoint \
  --device-map cuda:0 --max-steps 100 --output runs/chess-local.json
```

Use `--episodes N --seed N` for repeatable evaluations. Each episode is capped by
`--max-steps`; chess never invokes an oracle to rescue wasted key presses. The CLI
prints episode reward, steps, termination/truncation, and wall time, and saves every
completed episode. A trace is not a claim of gameplay competence. First HTTP inference
includes model loading; local CLI model loading happens before recording starts.

The compatibility example launchers also work:

```bash
uv run --locked --extra games python examples/chess/play.py --max-steps 20
uv run --locked --extra games python examples/doom/play.py --max-steps 20
```

## Common Gymnasium API

We use the maintained Gymnasium API rather than the legacy four-value Gym `step`:

```python
import gymnasium as gym
import litjev.games  # registers the environment IDs
from litjev.games.policy import HttpDecisionClient, LitJevPolicy

with gym.make("LitJev/ChessKeys-v0", max_steps=100) as env:
    policy = LitJevPolicy(
        HttpDecisionClient("http://127.0.0.1:8000"),
        env.unwrapped.action_names,
        env.unwrapped.instructions,
    )
    observation, info = env.reset(seed=7)
    while True:
        decision = policy.decide(observation)  # does NOT receive info
        observation, reward, terminated, truncated, info = env.step(decision.action)
        frame = env.render()
        if terminated or truncated:
            break
```

| Environment ID | Observation | Action space | Opponent / scenario |
| --- | --- | --- | --- |
| `LitJev/ChessKeys-v0` | RGB uint8, 480×640×3 | `Discrete(5)`: up, down, left, right, toggle | Seeded random opponent; player white by default |
| `LitJev/DoomButtons-v0` | RGB uint8, 480×640×3 | `Discrete(7)`: turn left/right, forward/backward, strafe left/right, attack | ViZDoom `deadly_corridor` by default |

Both implement `reset(seed, options)`, `step`, `render`, and `close`. `render_mode`
is `rgb_array` or `None`. Doom also supports `scenario="defend_the_center"` and
`resolution="160x120"`. Chess supports `player="black"`. Truncation is distinct
from game termination. The last real frame is returned if ViZDoom has no terminal
screen, with `info.terminal_frame_unavailable=true`.

Chess preserves the upstream geometric piece sprites, visible cursor, held-piece
indicator and legal-destination highlights. Their meanings are described in a fixed
instruction. The agent does not receive FEN, a legal-move list, the opponent's choice,
or a direct move action. Doom supplies no object labels, enemy coordinates, depth,
or telemetry to the policy. Environment `info` is for logging, not inference.

`LitJevPolicy` also works with another RGB Gymnasium environment when given 2–255
unique textual action descriptions matching its `Discrete` action indices. It maps
these names to Choice criteria keys. The shared compiler uses internal letter-code tokens;
the policy maps the returned original key back to an integer action.
It does not support continuous actions or change the environment's action semantics.

## Image API and readout

For a live Doom preview without any model or GPU inference:

```bash
uv run --extra games python -m litjev.games.preview --port 8012
```

Open `http://127.0.0.1:8012/` and press Play. Seven actions have equal probability.
A single background worker advances the environment; browsers subscribe through
SSE and never step it. Pause/reset/speed controls affect the shared session.
Episodes restart automatically. The worker continues when tabs are hidden or closed;
use Pause to stop it. Slow/disconnected viewers receive the latest state on reconnect,
not a lossless frame stream. Only 90 step summaries and the latest screenshot are
retained; the page shows a latency curve and the latest 48 actions.

For native ViZDoom recordings, add `--record runs/doom-demos`. Each process creates
a unique run subdirectory with one `.lmp` per episode; existing demos are never
overwritten. Files finalize when an episode ends or the server closes. Use short
relative paths (the native engine cannot reliably handle long demo paths).
Native demos are distinct from the LitJev JSON traces used by `litjev-film`.

These session/SSE, bounded timeline/latency and native-recording features adapt the
useful parts of [PR #2](https://github.com/zhengxuyu/litjev/pull/2) by xk into PR #4.
The symbolic health/ammo/actor-state policy, old enum schema and separate Doom page
from PR #2 are intentionally not imported: screenshots/Gym and the canonical Jev
schema remain the only model integration path.

This local preview is a shared single environment, not a multi-user model service.
Live preview and recorded replay use the same ported `film.html` layout and decision
renderer; random mode labels its uniform probabilities and does not fabricate logits.

`POST /v1/systemone/debug` accepts `{model, state, questions}` plus an optional
`image` field containing a PNG/JPEG
data URL or raw base64. It never fetches image URLs or server-side paths. Images are
limited to 4 MB encoded bytes and 1,048,576 pixels. The response is `{result, diagnostics}`.
`result` has the standard Jev response shape; logits, provenance and timings are in
`diagnostics`. The standard `/v1/systemone` endpoint rejects the image extension.

The processor builds the image/text prefix. Forward 1 runs Qwen vision and prefix
prefill; forward 2 reads candidate logits after each question's `Answer:` suffix. Its image
M-RoPE offset is carried into the cached suffix positions. This is two top-level
model forwards and one vision-tower call, with zero generated answer tokens.
Provenance includes image grid dimensions and the three readout rotary coordinates.

Every new screenshot requires a new prefill. Steps are causally sequential: the
next image exists only after the previous action. No claim of ten future decisions
in parallel, real-time Doom throughput, or calibrated action correctness is made.
Doom advances four game tics per decision; wall-clock inference can be much slower.

## Film / replay

Open **http://127.0.0.1:8000/film** and load a trace JSON (up to 100 MB), or build a
self-contained HTML replay you can open without a server:

```bash
uv run --locked --extra games litjev-film runs/chess.json --output runs/chess.html
uv run --locked --extra games litjev-film runs/doom.json --output runs/doom.html
```

The viewer shows the exact **pre-action image** alongside the selected button,
candidate distribution, raw logits, source position, latency, and **post-action**
reward/termination. Playback preserves recorded ordering and elapsed wall time;
the UI can explicitly accelerate playback. It does not pick only successful windows.
There are no invented attention maps or training curves: the upstream diagrams
describe a different, trained head and are deliberately not reused.

Optional silent MP4 export requires Node.js, Playwright's Chromium, and FFmpeg:

```bash
cd examples/film
npm ci
npx playwright install chromium
npm test
node render-film.mjs ../../runs/chess.html ../../runs/chess.mp4 10
```

The final argument caps the captured recorded wall time (default 60 seconds). Existing
MP4 outputs are not overwritten. Install FFmpeg separately and review its license.
No soundtrack is bundled. `runs/` and generated videos are ignored by Git.

## Trace contract

`schema_version="litjev.trace.v1"` identifies a trace with environment, model, seed,
action names, `training=false` and ordered `decisions`. Each decision contains:

- pre-action PNG data URL, episode/step, action name and integer index;
- logits/probabilities in action order, confidence, calibration status and provenance;
- observed policy latency, reported forward/output-token counts;
- post-action reward, cumulative reward, termination, truncation and diagnostic info.

`timestamp` is elapsed wall time from the episode start. The recorder never feeds
diagnostics back into the policy. Gym `info` may contain a chess move after execution
or Doom timing, but these do not enter the model's prompt. The film accepts this new
trace schema only, not upstream traces that assume trained-head activations.

## Verification

Current status: local tests (including real headless game environments and tiny
random Qwen image-cache equivalence) and the browser replay test pass. The actual
27B checkpoint processor has accepted a real chess screenshot. Full pretrained-Qwen
GPU gameplay and real-game MP4 export are still awaiting GPU allocation; no such
end-to-end result is claimed by this branch.

```bash
uv sync --locked --extra dev --extra games
uv run --no-sync pytest -q
uv run --no-sync ruff check .
# Optional real single-GPU smoke; loads the model once for both games:
uv run --no-sync python scripts/smoke_visual.py --model /path/to/checkpoint --steps 4
```

Tests cover the Gymnasium contract, deterministic seeds, real headless Doom, chess
keypresses/truncation, shared policy mapping, image validation, and trace escaping.
A tiny randomly initialized Qwen verifies that image-cache readouts match full
inputs, different pixels affect logits, and text behavior remains unchanged.
# Engine-assisted Doom text observations

Use `--observation engine_text` to feed Qwen a deterministic text description
instead of an image. This is a LitJev extension, not an upstream jevlike captioner.
ViZDoom label masks supply visible object names and horizontal screen positions;
depth-buffer pixels supply per-object and left/center/right median raw depths.
No captioning model, training, world coordinates, full map, or expert action rules
are used. At most 32 visible objects are listed, ordered by visible pixel area.
Depth values are raw 8-bit readings, not metric distances or navigability labels.

```sh
uv run --locked --extra games litjev-play doom --model /path/to/qwen-checkpoint \
  --observation engine_text --episodes 10 --max-steps 1000 --seed 0 \
  --output runs/doom-text.json
uv run --locked --extra games litjev-film runs/doom-text.json --output runs/doom-text.html
```

RGB remains the default and the Gym observation/render contract remains RGB.
Text-mode traces are explicitly marked `engine_labels_depth_text`, retain the
exact pre-action `observation_text`, and measure `description_ms` separately from
model `latency_ms`. Frames remain in the replay but are **not sent to the model**.
Compare the decision policies with care: this supplies engine-assisted semantics,
not equivalent information to a raw-pixel perception benchmark. Episode summaries
distinguish death, timeout, step limit and a live non-timeout scenario completion.
