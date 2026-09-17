from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litjev.schema import DecisionSchema


class McqQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    options: dict[str, str] = Field(min_length=2, max_length=10)


class McqRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[McqQuestion] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({q.question_id for q in self.questions}) != 10:
            raise ValueError("Question IDs must be unique")
        return self

    def to_schema(self):
        return DecisionSchema.from_mapping(
            {
                q.question_id: {
                    "type": "enum",
                    "choices": list(q.options),
                    "description": q.prompt
                    + "\n"
                    + "\n".join(f"{label}. {text}" for label, text in q.options.items()),
                }
                for q in self.questions
            }
        )


class SchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str
    schema_def: dict[str, dict] = Field(alias="schema", min_length=1, max_length=10)


def create_app(engine_factory):
    app = FastAPI(title="LitJev ten-question decisions")
    get_engine = lru_cache(maxsize=1)(engine_factory)
    load_lock = Lock()
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def playground():
        return FileResponse(static_dir / "index.html")

    def provide_engine():
        # lru_cache alone may execute its factory twice on concurrent cold requests.
        with load_lock:
            return get_engine()

    # Lets attached demos share this process's single loaded model.
    app.state.provide_engine = provide_engine

    def evaluate(state, schema):
        try:
            started = perf_counter()
            engine = provide_engine()
            ready = perf_counter()
            result = asdict(engine.decide(state, schema))
            finished = perf_counter()
            result["timing"] = {
                "model_setup_seconds": ready - started,
                "decision_seconds": finished - ready,
                "total_seconds": finished - started,
            }
            return result
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": get_engine.cache_info().currsize > 0}

    @app.post("/v1/batch-mcq")
    def batch_mcq(request: McqRequest):
        return evaluate("Answer each question using its listed options.", request.to_schema())

    @app.post("/v1/calibrated-schema")
    def schema_decision(request: SchemaRequest):
        try:
            schema = DecisionSchema.from_mapping(request.schema_def)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return evaluate(request.state, schema)

    return app
