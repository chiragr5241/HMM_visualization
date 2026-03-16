"""
Belief State Geometry — FastAPI backend
Serves the interactive MSP visualisation for MESS3, MESS3-2, RIVER, LEOPARD, FERN, FANIZZA.

Two computation modes:
  - "sample" (default): batch-sample sequences → fast, for live parameter exploration
  - "enumerate": full MSP tree enumeration → slow but complete

Run with:
    uvicorn app:app --port 8765 --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

_HERE = Path(__file__).parent

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
)
from simplexity.generative_processes.generalized_hidden_markov_model import (
    GeneralizedHiddenMarkovModel,
)
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.mixed_state_presentation import (
    LogMixedStateTreeGenerator,
    MixedStateTreeGenerator,
)

# ── Transition matrices from xavier/processes branch ──────────────────────────
# Source: github.com/Astera-org/simplexity/blob/xavier/processes/
#         simplexity/generative_processes/transition_matrices.py


def _leopard_matrix(x: float) -> jnp.ndarray:
    """Leopard HMM — 3 states, 2 symbols."""
    return jnp.array([
        [[0.0, 0.0, 0.3465],
         [0.6435, 0.0, 0.0],
         [0.0, 0.99 * x, 0.0]],
        [[0.005, 0.005, 0.6435],
         [0.3465, 0.005, 0.005],
         [0.005, 0.99 * (1.0 - x), 0.005]],
    ])


def _fern_matrix(x: float) -> jnp.ndarray:
    """Fern HMM — 3 states, 2 symbols."""
    return jnp.array([
        [[0.3942, 0.00512, 0.0381],
         [0.0, 0.53, 0.0],
         [0.0, 0.326 * x, 0.554]],
        [[0.3358, 0.01088, 0.2159],
         [0.0, 0.0, 0.47],
         [0.12, 0.326 * (1.0 - x), 0.0]],
    ])


# ── Precomputed constants ─────────────────────────────────────────────────────
_SQRT3_HALF = np.sqrt(3) / 2.0


# ── Utilities ─────────────────────────────────────────────────────────────────

def _barycentric(bs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map (N,3) probability vectors → 2D barycentric coords on the 2-simplex."""
    return bs[:, 1] + 0.5 * bs[:, 2], _SQRT3_HALF * bs[:, 2]


def _belief_to_rgb(bs: np.ndarray) -> np.ndarray:
    """Encode 3-state belief vectors as (N, 3) uint8 RGB (p0→R, p1→G, p2→B)."""
    return np.clip(bs[:, :3] * 255, 0, 255).astype(np.uint8)


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project (N, D) → 2D via PCA."""
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    X2d = Xc @ Vt[:2].T
    var_ratios = (S[:2] ** 2 / np.sum(S ** 2)).tolist()
    return X2d, var_ratios


# ── MSP tree enumeration (slow, complete) ─────────────────────────────────────

def _extract_hmm_msp(
    hmm: HiddenMarkovModel, max_seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    gen = LogMixedStateTreeGenerator(hmm, max_sequence_length=max_seq_len)
    tree = gen.generate()
    vals = list(tree.nodes.values())
    log_bs = jnp.array([v.log_belief_state for v in vals])
    log_probs = jnp.array([v.log_probability for v in vals])
    return np.array(jnp.exp(log_bs)), np.exp(np.array(log_probs))


def _extract_ghmm_msp(
    ghmm: GeneralizedHiddenMarkovModel,
    max_seq_len: int,
    prob_threshold: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    gen = MixedStateTreeGenerator(
        ghmm,
        max_sequence_length=max_seq_len,
        prob_threshold=prob_threshold,
    )
    tree = gen.generate()
    vals = list(tree.nodes.values())
    bs = jnp.array([v.belief_state for v in vals])
    probs = jnp.array([v.probability for v in vals])
    return np.array(bs), np.array(probs)


# ── Batch sampling (fast, for live updates) ───────────────────────────────────

def _sample_belief_states(
    model: HiddenMarkovModel | GeneralizedHiddenMarkovModel,
    batch_size: int,
    seq_len: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample batch_size sequences of seq_len steps, return all belief states.

    Returns (batch_size * seq_len, num_states) array of belief states.
    """
    init = jnp.tile(model.initial_state, (batch_size, 1))
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, batch_size)
    # states: (batch_size, seq_len, num_states)
    states, _obs = model.generate(init, keys, seq_len, True)
    # Include the initial state for each sequence
    # init is (batch_size, num_states) → (batch_size, 1, num_states)
    init_expanded = init[:, None, :]
    all_states = jnp.concatenate([init_expanded, states], axis=1)
    # Flatten to (batch_size * (seq_len+1), num_states)
    flat = all_states.reshape(-1, all_states.shape[-1])
    return np.array(flat)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Belief State Geometry", default_response_class=ORJSONResponse)
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_HERE / "static" / "index.html"))


class ComputeRequest(BaseModel):
    process: str
    params: dict[str, float] = {}
    mode: Literal["sample", "enumerate"] = "sample"
    # Sample mode params
    batch_size: int = 500
    seq_len: int = 20
    seed: int = 0
    # Enumerate mode params
    max_seq_len: int = 10


# ── Core computation ──────────────────────────────────────────────────────────

def _build_model(req: ComputeRequest):
    """Build the generative model from the request. Returns (model, is_ghmm)."""
    p = req.params
    name = req.process

    if name == "mess3":
        return build_hidden_markov_model("mess3", x=p.get("x", 0.15), a=p.get("a", 0.6)), False
    if name == "mess3_2":
        return build_hidden_markov_model(
            "mess3_2", x=p.get("x", 0.15), a=p.get("a", 0.6),
            p=p.get("p", 0.7), q=p.get("q", 0.3), r=p.get("r", 0.5),
        ), False
    if name == "river":
        return build_hidden_markov_model("river"), False
    if name == "leopard":
        return HiddenMarkovModel(_leopard_matrix(p.get("x", 0.5))), False
    if name == "fern":
        return HiddenMarkovModel(_fern_matrix(p.get("x", 0.5))), False
    if name == "fanizza":
        return build_generalized_hidden_markov_model(
            "fanizza", alpha=p.get("alpha", 2000.0), lamb=p.get("lamb", 0.49),
        ), True
    raise ValueError(f"Unknown process: {name}")


def _format_simplex(bs: np.ndarray) -> dict[str, Any]:
    """Format 3-state belief states for simplex plot."""
    bx, by = _barycentric(bs)
    rgb = _belief_to_rgb(bs)
    return {
        "type": "simplex",
        "x": bx.tolist(),
        "y": by.tolist(),
        "r": rgb[:, 0].tolist(),
        "g": rgb[:, 1].tolist(),
        "b": rgb[:, 2].tolist(),
        "bs": np.round(bs, 5).tolist(),
        "n": len(bs),
    }


def _format_pca(bs: np.ndarray, model) -> dict[str, Any]:
    """Format GHMM belief states for PCA plot."""
    obs_fn = eqx.filter_vmap(model.observation_probability_distribution)
    obs = np.array(obs_fn(jnp.array(bs)))
    p_obs0 = obs[:, 0]
    bs_2d, var_ratios = _pca_2d(bs)
    return {
        "type": "pca",
        "x": bs_2d[:, 0].tolist(),
        "y": bs_2d[:, 1].tolist(),
        "color_val": p_obs0.tolist(),
        "n": len(bs),
        "var_ratios": var_ratios,
    }


def _compute_sync(req: ComputeRequest) -> dict[str, Any]:
    model, is_ghmm = _build_model(req)

    if req.mode == "sample":
        bs = _sample_belief_states(model, req.batch_size, req.seq_len, req.seed)
        result = _format_pca(bs, model) if is_ghmm else _format_simplex(bs)
        result["mode"] = "sample"
        return result

    # enumerate mode
    if is_ghmm:
        bs, _probs = _extract_ghmm_msp(model, req.max_seq_len)
        result = _format_pca(bs, model)
    else:
        bs, _probs = _extract_hmm_msp(model, req.max_seq_len)
        result = _format_simplex(bs)
    result["mode"] = "enumerate"
    return result


@app.post("/api/compute")
async def compute(req: ComputeRequest) -> dict[str, Any]:
    return await run_in_threadpool(_compute_sync, req)
