"""FastAPI service (``api`` extra): ``POST /ask`` and ``GET /health``.

Run with ``gradguide serve`` or ``uvicorn gradguide.api:app``.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from gradguide.config import Settings
from gradguide.privacy.ratelimit import RateLimiter
from gradguide.retrieve.hybrid import MODES
from gradguide.service import GradGuide


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[tuple[str, str]] = Field(default_factory=list, max_length=20)
    mode: str | None = None


def create_app(service: GradGuide | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (service.settings if service else Settings.from_env())
    state: dict[str, GradGuide] = {}
    if service is not None:
        state["svc"] = service
    limiter = RateLimiter(settings.rate_limit_per_minute)
    app = FastAPI(title="gradguide", version="0.1.0")

    def svc() -> GradGuide:
        if "svc" not in state:  # built lazily, once per process
            state["svc"] = GradGuide(settings)
        return state["svc"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **svc().describe()}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> dict:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="Too many requests; try again in a minute.")
        if body.mode is not None and body.mode not in MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
        try:
            answer = svc().ask(body.question, body.history, body.mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return answer.to_dict()

    return app


def __getattr__(name: str):
    # `uvicorn gradguide.api:app` - build the default app only when it is requested
    if name == "app":
        return create_app()
    raise AttributeError(name)
