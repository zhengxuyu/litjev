"""Browser routes for the demo: the page, the two controls, and the decision event stream."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.responses import FileResponse, StreamingResponse

from litjev.doom.session import DoomSession, DoomSettings

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def attach_doom(app, engine_provider, settings: DoomSettings | None = None) -> DoomSession:
    session = DoomSession(engine_provider, settings)

    @app.get("/doom", include_in_schema=False)
    def page():
        return FileResponse(STATIC_DIR / "doom.html")

    @app.get("/doom/status")
    def status():
        return {"running": session.running, "scenario": session.settings.scenario}

    @app.post("/doom/start")
    def start():
        session.start()
        return {"running": session.running}

    @app.post("/doom/stop")
    def stop():
        session.stop()
        return {"running": session.running}

    @app.get("/doom/stream", include_in_schema=False)
    def stream():
        def events():
            for payload in session.watch():
                if payload is None:
                    yield ": keep-alive\n\n"
                else:
                    yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    app.state.doom_session = session
    return session
