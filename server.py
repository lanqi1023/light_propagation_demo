"""FastAPI server — serves static files and exposes /api/compute endpoint."""
import json
import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lightprop.types import Params
from lightprop.pipeline import compute


app = FastAPI(title="Light Propagation Demo")

ROOT = Path(__file__).parent


# ─── API ──────────────────────────────────────────────────────────────────────

class ComputeRequest(BaseModel):
    params: Params


@app.post("/api/compute")
def api_compute(req: ComputeRequest):
    result = compute(req.params)
    return JSONResponse({
        "intensity": _encode(result.intensity),
        "phase":     _encode(result.phase),
        "cross_x":   result.cross_x.tolist(),
        "cross_y":   result.cross_y.tolist(),
        "info":      result.info,
    })


@app.get("/api/fft-self-test")
def api_fft_self_test():
    from lightprop.fft import fft_self_test
    return JSONResponse(fft_self_test())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _encode(arr) -> str:
    """Encode ndarray as base64 JSON string."""
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ─── Static files ─────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    path = ROOT / "templates" / "index.html"
    return path.read_text(encoding="utf-8")
