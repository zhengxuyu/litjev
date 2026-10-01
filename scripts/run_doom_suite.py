# /// script
# requires-python = ">=3.12"
# dependencies = ["vizdoom==1.3.0", "numpy==2.5.3", "pydantic==2.13.5", "typesafe-sdk==0.7.0", "httpx>=0.28"]
# ///
"""Run with uv run scripts/run_doom_suite.py --backend jev --output /absolute/path."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from litjev.games.suite.runner import main

if __name__ == '__main__':
    main()
