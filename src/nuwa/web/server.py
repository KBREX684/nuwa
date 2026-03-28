"""FastAPI backend for Nuwa's real-time training dashboard.

Provides REST endpoints for configuration, training control, SSE event
streaming, and result inspection.  A demo mode is included for UI testing
without real API keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nuwa.config.schema import NuwaConfig
from nuwa.core.types import (
    RoundResult,
    TrainingConfig,
    TrainingResult,
)
from nuwa.engine.loop import TrainingLoop
from nuwa.guardrails.consistency import ConsistencyGuardrail
from nuwa.guardrails.overfitting import OverfittingGuardrail
from nuwa.guardrails.regression import RegressionGuardrail
from nuwa.llm.backend import LiteLLMBackend
from nuwa.persistence.run_log import RunLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSE dependency (sse-starlette)
# ---------------------------------------------------------------------------

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover
    EventSourceResponse = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# App-level mutable state
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "current_config": None,          # NuwaConfig | None
    "training_status": "idle",        # idle | running | completed | error
    "training_result": None,          # TrainingResult | None
    "round_history": [],              # list[dict]  (round event dicts for SSE)
    "current_round": 0,
    "current_stage": "",
    "error_message": None,            # str | None
    "stop_requested": False,
    "event_queue": None,              # asyncio.Queue | None
    "training_task": None,            # asyncio.Task | None
}

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ConfigRequest(BaseModel):
    """POST /api/config request body."""
    llm_model: str = "openai/gpt-4o"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    connector_type: str = "function"
    connector_params: dict[str, Any] = Field(default_factory=dict)
    training_direction: str = "Improve the target agent's response quality."
    max_rounds: int = 10
    samples_per_round: int = 20
    train_val_split: float = 0.7
    overfitting_threshold: float = 0.15
    regression_tolerance: float = 0.05
    consistency_threshold: float = 0.8


class ApproveRequest(BaseModel):
    """POST /api/approve request body."""
    decision: str  # "accept" | "reject" | "extend"
    extra_rounds: int = 5


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Nuwa Training Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _on_startup() -> None:
    """Create the .nuwa project directory on startup."""
    project_dir = Path(".nuwa")
    project_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Nuwa project directory ensured at %s", project_dir.resolve())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_state() -> None:
    """Reset volatile training state for a new run."""
    _state["training_status"] = "idle"
    _state["training_result"] = None
    _state["round_history"] = []
    _state["current_round"] = 0
    _state["current_stage"] = ""
    _state["error_message"] = None
    _state["stop_requested"] = False
    _state["event_queue"] = asyncio.Queue()
    _state["training_task"] = None


def _push_event(event: dict[str, Any]) -> None:
    """Enqueue an SSE event dict (non-blocking)."""
    q: asyncio.Queue | None = _state.get("event_queue")
    if q is not None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("SSE event queue full; dropping event")


def _result_to_json(result: TrainingResult) -> dict[str, Any]:
    """Serialise a TrainingResult to a JSON-safe dict."""
    return json.loads(result.model_dump_json())


# ---------------------------------------------------------------------------
# Endpoint: GET /api/status
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Return the current dashboard state."""
    return JSONResponse({
        "training_status": _state["training_status"],
        "current_round": _state["current_round"],
        "current_stage": _state["current_stage"],
        "error": _state["error_message"],
    })


# ---------------------------------------------------------------------------
# Endpoint: POST /api/config
# ---------------------------------------------------------------------------


@app.post("/api/config")
async def post_config(body: ConfigRequest) -> JSONResponse:
    """Validate and store a NuwaConfig from the provided parameters."""
    try:
        cfg = NuwaConfig(
            llm_model=body.llm_model,
            llm_api_key=body.llm_api_key,
            llm_base_url=body.llm_base_url,
            connector_type=body.connector_type,
            connector_params=body.connector_params,
            training_direction=body.training_direction,
            max_rounds=body.max_rounds,
            samples_per_round=body.samples_per_round,
            train_val_split=body.train_val_split,
            overfitting_threshold=body.overfitting_threshold,
            regression_tolerance=body.regression_tolerance,
            consistency_threshold=body.consistency_threshold,
        )
        _state["current_config"] = cfg
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


# ---------------------------------------------------------------------------
# Endpoint: POST /api/train/start
# ---------------------------------------------------------------------------


def _make_round_callback(backend: Any) -> Any:
    """Return a callback that emits SSE events after each round."""

    def _callback(round_result: RoundResult, context: Any) -> None:
        train_mean = round_result.train_scores.mean_score
        val_mean = (
            round_result.val_scores.mean_score if round_result.val_scores else 0.0
        )
        reflection_text = (
            round_result.reflection.diagnosis if round_result.reflection else ""
        )
        mutation_text = (
            round_result.mutation.description if round_result.mutation else ""
        )
        event: dict[str, Any] = {
            "type": "round_end",
            "round": round_result.round_num,
            "train_score": round(train_mean, 4),
            "val_score": round(val_mean, 4),
            "reflection": reflection_text,
            "mutation": mutation_text,
            "applied": round_result.applied,
        }
        _state["round_history"].append(event)
        _state["current_round"] = round_result.round_num
        _push_event(event)

        # Advance mock round hint if applicable
        if hasattr(backend, "_round_hint"):
            backend._round_hint = round_result.round_num + 1

    return _callback


async def _run_training(
    config: NuwaConfig,
    backend: Any,
    target: Any,
) -> None:
    """Execute the training loop in the background, emitting SSE events."""
    training_config = config.build_training_config()

    guardrails = [
        OverfittingGuardrail(threshold=config.overfitting_threshold),
        RegressionGuardrail(tolerance=config.regression_tolerance),
        ConsistencyGuardrail(threshold=config.consistency_threshold),
    ]

    callback = _make_round_callback(backend)

    loop = TrainingLoop(
        config=training_config,
        backend=backend,
        target=target,
        guardrails=guardrails,
        callbacks=[callback],
    )

    _state["training_status"] = "running"

    # Emit initial event
    _push_event({
        "type": "round_start",
        "round": 1,
        "max_rounds": training_config.max_rounds,
    })

    try:
        # Wrap the loop to inject per-round stage events and stop check.
        # The TrainingLoop itself drives stages internally, so we hook via
        # the callback registered above.  We run the loop directly.
        result: TrainingResult = await loop.run()

        _state["training_result"] = result
        _state["training_status"] = "completed"

        _push_event({
            "type": "training_complete",
            "best_round": result.best_round,
            "best_score": round(result.best_val_score, 4),
            "stop_reason": result.stop_reason,
        })

    except Exception as exc:
        logger.exception("Training failed")
        _state["training_status"] = "error"
        _state["error_message"] = str(exc)
        _push_event({"type": "error", "message": str(exc)})


@app.post("/api/train/start")
async def train_start() -> JSONResponse:
    """Start a training run in a background asyncio task."""
    if _state["training_status"] == "running":
        return JSONResponse(
            {"error": "Training is already running."}, status_code=409,
        )

    cfg: NuwaConfig | None = _state.get("current_config")  # type: ignore[assignment]
    if cfg is None:
        return JSONResponse(
            {"error": "No configuration set. POST /api/config first."},
            status_code=400,
        )

    _reset_state()
    _state["current_config"] = cfg  # preserve config after reset

    # Build real backend and connector
    try:
        llm_kwargs = cfg.build_llm_kwargs()
        backend = LiteLLMBackend(**llm_kwargs)
        target = cfg.build_connector()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    task = asyncio.create_task(_run_training(cfg, backend, target))
    _state["training_task"] = task
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Endpoint: POST /api/train/stop
# ---------------------------------------------------------------------------


@app.post("/api/train/stop")
async def train_stop() -> JSONResponse:
    """Request the training loop to stop after the current round."""
    _state["stop_requested"] = True
    task: asyncio.Task | None = _state.get("training_task")
    if task is not None and not task.done():
        task.cancel()
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Endpoint: GET /api/train/events  (SSE)
# ---------------------------------------------------------------------------


async def _event_generator(request: Request):
    """Yield SSE events from the queue with keepalive pings."""
    q: asyncio.Queue = _state.get("event_queue") or asyncio.Queue()
    if _state.get("event_queue") is None:
        _state["event_queue"] = q

    while True:
        if await request.is_disconnected():
            break
        try:
            event = await asyncio.wait_for(q.get(), timeout=15.0)
            yield {"data": json.dumps(event)}
        except asyncio.TimeoutError:
            # Keepalive ping
            yield {"event": "ping", "data": ""}
        except asyncio.CancelledError:
            break


@app.get("/api/train/events")
async def train_events(request: Request):
    """Server-Sent Events endpoint for real-time training updates."""
    if EventSourceResponse is None:
        return JSONResponse(
            {"error": "sse-starlette is not installed."},
            status_code=501,
        )
    return EventSourceResponse(_event_generator(request))


# ---------------------------------------------------------------------------
# Endpoint: GET /api/results
# ---------------------------------------------------------------------------


@app.get("/api/results")
async def get_results() -> JSONResponse:
    """Return the full TrainingResult, or null if unavailable."""
    result: TrainingResult | None = _state.get("training_result")
    if result is None:
        return JSONResponse(None)
    return JSONResponse(_result_to_json(result))


# ---------------------------------------------------------------------------
# Endpoint: GET /api/results/rounds
# ---------------------------------------------------------------------------


@app.get("/api/results/rounds")
async def get_results_rounds() -> JSONResponse:
    """Return round-by-round data suitable for charting."""
    result: TrainingResult | None = _state.get("training_result")
    if result is None:
        return JSONResponse([])

    rounds_data: list[dict[str, Any]] = []
    for rr in result.rounds:
        train_mean = rr.train_scores.mean_score
        val_mean = rr.val_scores.mean_score if rr.val_scores else None
        rounds_data.append({
            "round": rr.round_num,
            "train_score": round(train_mean, 4),
            "val_score": round(val_mean, 4) if val_mean is not None else None,
            "train_pass_rate": round(rr.train_scores.pass_rate, 4),
            "val_pass_rate": (
                round(rr.val_scores.pass_rate, 4) if rr.val_scores else None
            ),
            "reflection": (
                rr.reflection.diagnosis if rr.reflection else ""
            ),
            "mutation": (
                rr.mutation.description if rr.mutation else ""
            ),
            "applied": rr.applied,
            "timestamp": rr.timestamp.isoformat(),
        })
    return JSONResponse(rounds_data)


# ---------------------------------------------------------------------------
# Endpoint: POST /api/approve
# ---------------------------------------------------------------------------


@app.post("/api/approve")
async def post_approve(body: ApproveRequest) -> JSONResponse:
    """Handle human-in-the-loop decision on training results."""
    result: TrainingResult | None = _state.get("training_result")

    if body.decision == "accept":
        if result is None:
            return JSONResponse(
                {"error": "No training result to accept."}, status_code=400,
            )
        # Save the final config to disk
        project_dir = Path(".nuwa")
        project_dir.mkdir(parents=True, exist_ok=True)
        config_path = project_dir / "accepted_config.json"
        config_path.write_text(
            json.dumps(result.final_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Accepted config saved to %s", config_path)
        return JSONResponse({"ok": True})

    elif body.decision == "reject":
        _state["training_result"] = None
        _state["training_status"] = "idle"
        return JSONResponse({"ok": True})

    elif body.decision == "extend":
        cfg: NuwaConfig | None = _state.get("current_config")  # type: ignore[assignment]
        if cfg is None:
            return JSONResponse(
                {"error": "No configuration available."}, status_code=400,
            )
        if _state["training_status"] == "running":
            return JSONResponse(
                {"error": "Training is already running."}, status_code=409,
            )
        # Update max_rounds and restart
        extended_cfg = cfg.model_copy(
            update={"max_rounds": body.extra_rounds},
        )
        _state["current_config"] = extended_cfg
        _reset_state()
        _state["current_config"] = extended_cfg

        try:
            llm_kwargs = extended_cfg.build_llm_kwargs()
            backend = LiteLLMBackend(**llm_kwargs)
            target = extended_cfg.build_connector()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        task = asyncio.create_task(_run_training(extended_cfg, backend, target))
        _state["training_task"] = task
        return JSONResponse({"ok": True})

    else:
        return JSONResponse(
            {"error": f"Unknown decision: {body.decision!r}"},
            status_code=400,
        )


# ---------------------------------------------------------------------------
# Endpoint: GET /api/history
# ---------------------------------------------------------------------------


@app.get("/api/history")
async def get_history() -> JSONResponse:
    """Return past training rounds from the RunLog on disk."""
    try:
        run_log = RunLog(Path(".nuwa") / "logs")
        history = run_log.load_history()
        return JSONResponse([
            json.loads(rr.model_dump_json()) for rr in history
        ])
    except Exception as exc:
        logger.exception("Failed to load history")
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Demo mode: POST /api/demo/start
# ---------------------------------------------------------------------------


async def _run_demo_training() -> None:
    """Run a mock training loop with simulated delays for UI testing."""
    import random

    from nuwa.core.types import (
        AgentResponse,
        EvalSample,
        Mutation,
        Reflection,
        ScoreCard,
        ScoredResult,
    )

    _state["training_status"] = "running"
    max_rounds = 5
    rounds_results: list[RoundResult] = []

    _push_event({
        "type": "round_start",
        "round": 1,
        "max_rounds": max_rounds,
    })

    try:
        for round_num in range(1, max_rounds + 1):
            if _state.get("stop_requested"):
                break

            _state["current_round"] = round_num

            # -- Stage: dataset_gen --
            _state["current_stage"] = "dataset_gen"
            _push_event({
                "type": "stage",
                "round": round_num,
                "stage": "dataset_gen",
            })
            await asyncio.sleep(0.5)

            # -- Stage: execution --
            _state["current_stage"] = "execution"
            _push_event({
                "type": "stage",
                "round": round_num,
                "stage": "execution",
            })
            await asyncio.sleep(0.5)

            # -- Stage: evaluation --
            _state["current_stage"] = "evaluation"
            _push_event({
                "type": "stage",
                "round": round_num,
                "stage": "evaluation",
            })
            await asyncio.sleep(0.5)

            # -- Stage: reflection --
            _state["current_stage"] = "reflection"
            _push_event({
                "type": "stage",
                "round": round_num,
                "stage": "reflection",
            })
            await asyncio.sleep(0.5)

            # -- Stage: mutation --
            _state["current_stage"] = "mutation"
            _push_event({
                "type": "stage",
                "round": round_num,
                "stage": "mutation",
            })
            await asyncio.sleep(0.5)

            # Build simulated scores (improving over rounds)
            base_train = 0.45 + round_num * 0.08 + random.uniform(-0.03, 0.03)
            base_val = base_train - random.uniform(0.02, 0.06)

            sample = EvalSample(
                input_text="Demo question",
                expected_behavior="A good answer",
                difficulty="medium",
            )
            response = AgentResponse(
                output_text="Demo response",
                latency_ms=120.0,
            )

            train_results = [
                ScoredResult(
                    sample=sample,
                    response=response,
                    score=max(0.0, min(1.0, base_train + random.uniform(-0.1, 0.1))),
                    reasoning="Demo scoring",
                )
                for _ in range(7)
            ]
            val_results = [
                ScoredResult(
                    sample=sample,
                    response=response,
                    score=max(0.0, min(1.0, base_val + random.uniform(-0.1, 0.1))),
                    reasoning="Demo scoring",
                )
                for _ in range(3)
            ]

            train_card = ScoreCard(
                results=train_results,
                failure_analysis="Demo failure analysis",
            )
            val_card = ScoreCard(
                results=val_results,
                failure_analysis="Demo val analysis",
            )

            reflection = Reflection(
                round_num=round_num,
                diagnosis=f"Round {round_num}: The agent improved on detail but still misses edge cases.",
                failure_patterns=["Insufficient examples", "Missing edge cases"],
                proposed_changes=["Add explicit example instructions"],
                priority="medium",
            )

            mutation = Mutation(
                description=f"Enhance system prompt with detail-level=high for round {round_num}",
                original_config={"detail_level": "medium"},
                proposed_config={"detail_level": "high"},
                reasoning="Addresses insufficient detail feedback.",
            )

            rr = RoundResult(
                round_num=round_num,
                train_scores=train_card,
                val_scores=val_card,
                reflection=reflection,
                mutation=mutation,
                applied=True,
            )
            rounds_results.append(rr)

            # Emit round_end event
            event: dict[str, Any] = {
                "type": "round_end",
                "round": round_num,
                "train_score": round(train_card.mean_score, 4),
                "val_score": round(val_card.mean_score, 4),
                "reflection": reflection.diagnosis,
                "mutation": mutation.description,
                "applied": True,
            }
            _state["round_history"].append(event)
            _push_event(event)

        # Determine best round
        best_round = 0
        best_score = 0.0
        for rr in rounds_results:
            vs = rr.val_scores.mean_score if rr.val_scores else 0.0
            if vs > best_score:
                best_score = vs
                best_round = rr.round_num

        stop_reason = (
            "stopped by user" if _state.get("stop_requested") else "max_rounds reached"
        )

        result = TrainingResult(
            rounds=rounds_results,
            best_round=best_round,
            best_val_score=round(best_score, 4),
            final_config={"system_prompt": "You are a helpful assistant.", "detail_level": "high"},
            stop_reason=stop_reason,
            total_duration_s=round(max_rounds * 2.5, 2),
        )

        _state["training_result"] = result
        _state["training_status"] = "completed"
        _push_event({
            "type": "training_complete",
            "best_round": result.best_round,
            "best_score": round(result.best_val_score, 4),
            "stop_reason": result.stop_reason,
        })

    except asyncio.CancelledError:
        _state["training_status"] = "error"
        _state["error_message"] = "Demo training cancelled."
        _push_event({"type": "error", "message": "Demo training cancelled."})
    except Exception as exc:
        logger.exception("Demo training failed")
        _state["training_status"] = "error"
        _state["error_message"] = str(exc)
        _push_event({"type": "error", "message": str(exc)})


@app.post("/api/demo/start")
async def demo_start() -> JSONResponse:
    """Start a mock training run for UI testing without real API keys."""
    if _state["training_status"] == "running":
        return JSONResponse(
            {"error": "Training is already running."}, status_code=409,
        )

    _reset_state()

    task = asyncio.create_task(_run_demo_training())
    _state["training_task"] = task
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Static file serving (SPA fallback)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def _spa_fallback(full_path: str) -> Any:
    """Serve static files with SPA index.html fallback."""
    from starlette.responses import FileResponse

    file_path = _STATIC_DIR / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({"error": "Not found"}, status_code=404)


# Try to mount static files as well (for auto directory listings / efficient serving)
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# __main__ entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
