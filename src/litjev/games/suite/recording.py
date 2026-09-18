"""Full-rate video plus lossless PCM, independently of API wall-clock latency."""

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, directory, width=640, height=480):
        self.directory = Path(directory)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for recording")
        self.ffmpeg = ffmpeg
        self.errors = (self.directory / "ffmpeg.log").open("wb")
        self.video = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                "35",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(self.directory / "video.mp4"),
            ],
            stdin=subprocess.PIPE,
            stderr=self.errors,
        )
        self.audio = wave.open(str(self.directory / "audio.wav"), "wb")  # noqa: SIM115 - closed by close()
        self.audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        self.frames = 0
        self.repeated_terminal_frames = 0
        self.ledger = (self.directory / "frames.jsonl").open("w")

    def append(self, frame, audio, game_tic, decision, terminal=False):
        self.video.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        pcm = np.asarray(audio, dtype="<i2")
        if pcm.shape != (1260, 2):
            raise ValueError(f"Unexpected per-tic audio shape: {pcm.shape}")
        self.audio.writeframesraw(pcm.tobytes())
        self.ledger.write(
            json.dumps(
                {
                    "frame": self.frames,
                    "video_seconds": self.frames / 35,
                    "game_tic": game_tic,
                    "decision": decision,
                    "terminal_frame_repeated": terminal,
                }
            )
            + "\n"
        )
        self.frames += 1
        self.repeated_terminal_frames += int(terminal)

    def close(self):
        self.video.stdin.close()
        self.audio.close()
        self.ledger.close()
        code = self.video.wait(timeout=60)
        self.errors.close()
        if code:
            raise RuntimeError("Video encoding failed; see ffmpeg.log")
        subprocess.run(
            [
                self.ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self.directory / "video.mp4"),
                "-i",
                str(self.directory / "audio.wav"),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(self.directory / "replay.mp4"),
            ],
            check=True,
            timeout=60,
        )
        return {
            "frames": self.frames,
            "fps": 35,
            "seconds": self.frames / 35,
            "terminal_frames_repeated": self.repeated_terminal_frames,
            "video": "replay.mp4",
            "lossless_audio": "audio.wav",
            "demo": "episode.lmp",
        }
