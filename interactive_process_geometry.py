"""Interactive Dash app for exploring 3-state, 2-token HMM process geometry.

Provides 15 sliders (6 for row-stochastic s, 9 for mixing t) and three
synchronized 2D scatter plots: belief states, observation probabilities,
and log observation probabilities.

Usage:
    uv run python -m experiments.analysis.process.interactive_process_geometry
"""

# Workaround: x86 Python on ARM Mac (Rosetta) fails JAX's AVX check.
try:
    import jaxlib.cpu_feature_guard as _jax_cpu_guard
    _jax_cpu_guard.check_cpu_features = lambda: None
except Exception:
    pass

import numpy as np
import jax.numpy as jnp
from dash import Dash, Input, Output, callback, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_V1 = np.array([1, -1, 0]) / np.sqrt(2)
_V2 = np.array([-1, -1, 2]) / np.sqrt(6)
_SIMPLEX_VERTICES = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_EPS = 1e-12

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def three_state_two_token(
    s: np.ndarray,
    t: np.ndarray,
) -> jnp.ndarray:
    """Parameterise two 3x3 row-substochastic matrices whose sum is row-stochastic.

    Args:
        s: (3, 3) row-stochastic matrix (total transition structure).
        t: (3, 3) mixing matrix with entries in [0, 1].

    Returns:
        (2, 3, 3) JAX array where [0] = T . S and [1] = (1 - T) . S.
    """
    s = jnp.asarray(s)
    t = jnp.asarray(t)
    a = t * s
    b = (1 - t) * s
    return jnp.stack([a, b])


def stick_breaking_to_stochastic(params: list[float]) -> np.ndarray:
    """Convert 6 params in [0, 1] to a 3x3 row-stochastic matrix.

    For each row *i* with parameters (a_i, b_i):
        s[i] = (a_i, b_i * (1 - a_i), (1 - a_i) * (1 - b_i))
    """
    s = np.zeros((3, 3))
    for i in range(3):
        a_i = params[2 * i]
        b_i = params[2 * i + 1]
        s[i, 0] = a_i
        s[i, 1] = b_i * (1 - a_i)
        s[i, 2] = (1 - a_i) * (1 - b_i)
    return s


def project_beliefs_to_2d(beliefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-simplex belief states to 2D via centred simplex coordinates."""
    center = np.ones(3) / 3
    centered = beliefs - center
    return centered @ _V1, centered @ _V2


def _format_seq(seq: tuple[int, ...]) -> str:
    if len(seq) == 0:
        return "()"
    return "(" + ", ".join(str(s) for s in seq) + ")"


def enumerate_beliefs_and_obs(
    hmm: HiddenMarkovModel,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Enumerate unique belief states and compute observation probabilities.

    Returns:
        beliefs: [n, 3]
        obs_probs: [n, 2]
        log_obs_probs: [n, 2]
        min_depths: [n]
        counts: [n] number of sequences mapping to each unique belief
        sequences: list of shortest observation sequence per unique belief
    """
    tree = MixedStateTreeGenerator(hmm, max_sequence_length=max_seq_len).generate()

    seen: dict[tuple[float, ...], int] = {}
    beliefs_list: list[np.ndarray] = []
    min_depths_list: list[int] = []
    counts_list: list[int] = []
    sequences_list: list[tuple[int, ...]] = []

    for seq, node in tree.nodes.items():
        if node.probability <= 0:
            continue
        b_key = tuple(round(float(b), 8) for b in node.belief_state)
        depth = len(seq)
        if b_key not in seen:
            seen[b_key] = len(beliefs_list)
            beliefs_list.append(np.array(node.belief_state, dtype=np.float64))
            min_depths_list.append(depth)
            counts_list.append(1)
            sequences_list.append(seq)
        else:
            idx = seen[b_key]
            counts_list[idx] += 1
            if depth < min_depths_list[idx]:
                min_depths_list[idx] = depth
                sequences_list[idx] = seq

    beliefs = np.array(beliefs_list)
    min_depths = np.array(min_depths_list)
    counts = np.array(counts_list)

    obs_probs_list = []
    for b in beliefs_list:
        op = hmm.observation_probability_distribution(jnp.array(b))
        obs_probs_list.append(np.array(op, dtype=np.float64))
    obs_probs = np.array(obs_probs_list)
    log_obs_probs = np.log(obs_probs + _EPS)

    return beliefs, obs_probs, log_obs_probs, min_depths, counts, sequences_list


def compute_colors(
    beliefs: np.ndarray,
    obs_probs: np.ndarray,
    log_obs_probs: np.ndarray,
    min_depths: np.ndarray,
    color_mode: str,
) -> list[str] | np.ndarray:
    """Compute per-point colours based on the selected mode."""
    if color_mode == "belief":
        return [
            f"rgb({int(b[0] * 255)},{int(b[1] * 255)},{int(b[2] * 255)})"
            for b in beliefs
        ]
    if color_mode == "obs_prob":
        return [
            f"rgb({int(p[0] * 255)},{int(p[1] * 255)},128)"
            for p in obs_probs
        ]
    if color_mode == "log_obs_prob":
        lo, hi = log_obs_probs.min(), log_obs_probs.max()
        span = hi - lo if hi - lo > 1e-10 else 1.0
        normed = (log_obs_probs - lo) / span
        return [
            f"rgb({int(n[0] * 255)},{int(n[1] * 255)},128)"
            for n in normed
        ]
    # depth
    return min_depths


def build_hmm(
    s_params: list[float],
    t_params: list[float],
    use_stationary: bool,
    init_params: list[float] | None = None,
) -> HiddenMarkovModel:
    """Build an HMM from slider values."""
    s = stick_breaking_to_stochastic(s_params)
    t = np.array(t_params).reshape(3, 3)
    transition_matrices = three_state_two_token(s, t)

    initial_state = None
    if not use_stationary and init_params is not None:
        a, b = init_params
        initial_state = jnp.array([a, b * (1 - a), (1 - a) * (1 - b)])

    return HiddenMarkovModel(transition_matrices, initial_state=initial_state)


def compute_kernel_direction(transition_matrices: np.ndarray) -> np.ndarray | None:
    """Compute the direction in belief space that maps to zero under the belief-to-obs-prob map.

    The linear map from beliefs to observation probabilities is b → b @ M,
    where M[s, x] = sum_{s'} T[x, s, s']. Restricted to the simplex tangent
    space (vectors summing to 0), this maps R^2 → R^1, so its kernel is
    generically one-dimensional.

    Returns the kernel direction (summing to 0, unit norm) or None if degenerate.
    """
    m = np.array(transition_matrices).sum(axis=2).T  # (num_states, vocab_size)
    v = np.cross(np.ones(3), m[:, 0])
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    return v / norm


def _kernel_simplex_endpoints(v: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the two points where the line centroid + t*v intersects the simplex boundary."""
    c = np.array([1 / 3, 1 / 3, 1 / 3])
    t_min_val = -np.inf
    t_max_val = np.inf
    for i in range(3):
        if abs(v[i]) < 1e-15:
            continue
        bound = -c[i] / v[i]
        if v[i] > 0:
            t_min_val = max(t_min_val, bound)
        else:
            t_max_val = min(t_max_val, bound)
    if t_min_val >= t_max_val:
        return None
    return c + t_min_val * v, c + t_max_val * v


def _simplex_triangle() -> tuple[np.ndarray, np.ndarray]:
    """Return x, y arrays tracing the projected simplex triangle (closed)."""
    verts_2d = np.column_stack(project_beliefs_to_2d(_SIMPLEX_VERTICES))
    closed = np.vstack([verts_2d, verts_2d[:1]])
    return closed[:, 0], closed[:, 1]


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def _slider(id_: str, label: str, value: float = 0.5) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "12px"}),
        dcc.Slider(
            id=id_, min=0, max=1, step=0.01, value=value,
            marks=None, tooltip={"placement": "right", "always_visible": False},
        ),
    ], style={"marginBottom": "2px"})


def _build_layout() -> html.Div:
    s_sliders = html.Div([
        html.H4("s matrix (row-stochastic)", style={"marginBottom": "4px"}),
        html.P(
            "Each row: s[i] = (a, b(1-a), (1-a)(1-b))",
            style={"fontSize": "11px", "color": "#666", "margin": "0 0 6px 0"},
        ),
        *[
            html.Div([
                html.Label(f"Row {i}", style={"fontWeight": "bold", "fontSize": "12px"}),
                _slider(f"s-a{i}", f"a{i} → P(→S0)", value=0.33),
                _slider(f"s-b{i}", f"b{i} → split S1/S2", value=0.50),
            ], style={"marginBottom": "8px"})
            for i in range(3)
        ],
    ])

    t_sliders = html.Div([
        html.H4("t matrix (mixing)", style={"marginBottom": "4px"}),
        html.P(
            "t[i,j] ∈ [0,1]: fraction of s[i,j] assigned to obs 0",
            style={"fontSize": "11px", "color": "#666", "margin": "0 0 6px 0"},
        ),
        *[
            html.Div([
                html.Label(f"Row {i}", style={"fontWeight": "bold", "fontSize": "12px"}),
                *[_slider(f"t-{i}{j}", f"t[{i},{j}]") for j in range(3)],
            ], style={"marginBottom": "8px"})
            for i in range(3)
        ],
    ])

    controls = html.Div([
        html.H4("Controls", style={"marginBottom": "4px"}),
        html.Button(
            "Randomise", id="randomise-btn",
            style={"width": "100%", "marginBottom": "8px", "padding": "6px"},
        ),
        html.Label("Max sequence length", style={"fontSize": "12px"}),
        dcc.Dropdown(
            id="max-seq-len",
            options=[{"label": str(n), "value": n} for n in range(1, 11)],
            value=6, clearable=False,
            style={"marginBottom": "8px"},
        ),
        html.Label("Color mode", style={"fontSize": "12px"}),
        dcc.Dropdown(
            id="color-mode",
            options=[
                {"label": "Belief coordinates", "value": "belief"},
                {"label": "Obs prob coordinates", "value": "obs_prob"},
                {"label": "Log obs prob coordinates", "value": "log_obs_prob"},
                {"label": "Sequence length", "value": "depth"},
            ],
            value="belief", clearable=False,
            style={"marginBottom": "8px"},
        ),
        dcc.Checklist(
            id="use-stationary",
            options=[{"label": " Use stationary initial distribution", "value": "on"}],
            value=["on"],
            style={"marginBottom": "8px"},
        ),
        html.Div(id="init-sliders-container", children=[
            html.Label("Initial distribution (stick-breaking)", style={"fontSize": "12px"}),
            _slider("init-a", "a → P(S0)", value=0.33),
            _slider("init-b", "b → split S1/S2", value=0.50),
        ]),
    ])

    info_panel = html.Div([
        html.H4("Info"),
        html.Pre(id="info-panel", style={"fontSize": "11px", "whiteSpace": "pre-wrap"}),
    ])

    sidebar = html.Div(
        [s_sliders, html.Hr(), t_sliders, html.Hr(), controls, html.Hr(), info_panel],
        style={
            "width": "350px", "padding": "10px", "overflowY": "auto",
            "borderRight": "1px solid #ddd", "flexShrink": "0", "height": "100vh",
        },
    )

    main = html.Div(
        [dcc.Graph(id="main-figure", style={"height": "100%", "width": "100%"})],
        style={"flex": "1", "height": "100vh"},
    )

    return html.Div(
        [sidebar, main],
        style={"display": "flex", "fontFamily": "sans-serif"},
    )


app = Dash(__name__)
app.layout = _build_layout()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("init-sliders-container", "style"),
    Input("use-stationary", "value"),
)
def toggle_init_sliders(use_stationary: list[str]) -> dict:
    """Show/hide initial distribution sliders."""
    if "on" in (use_stationary or []):
        return {"display": "none"}
    return {"display": "block"}


_SLIDER_IDS = [
    "s-a0", "s-b0", "s-a1", "s-b1", "s-a2", "s-b2",
    "t-00", "t-01", "t-02", "t-10", "t-11", "t-12", "t-20", "t-21", "t-22",
]


@callback(
    *(Output(sid, "value") for sid in _SLIDER_IDS),
    Input("randomise-btn", "n_clicks"),
    prevent_initial_call=True,
)
def randomise_sliders(_n_clicks: int | None) -> tuple[float, ...]:
    """Sample all 15 parameters uniformly from [0, 1]."""
    rng = np.random.default_rng()
    values = rng.uniform(0, 1, size=len(_SLIDER_IDS))
    return tuple(round(float(v), 2) for v in values)


@callback(
    Output("main-figure", "figure"),
    Output("info-panel", "children"),
    # s params
    Input("s-a0", "value"), Input("s-b0", "value"),
    Input("s-a1", "value"), Input("s-b1", "value"),
    Input("s-a2", "value"), Input("s-b2", "value"),
    # t params
    Input("t-00", "value"), Input("t-01", "value"), Input("t-02", "value"),
    Input("t-10", "value"), Input("t-11", "value"), Input("t-12", "value"),
    Input("t-20", "value"), Input("t-21", "value"), Input("t-22", "value"),
    # controls
    Input("max-seq-len", "value"),
    Input("use-stationary", "value"),
    Input("init-a", "value"),
    Input("init-b", "value"),
    Input("color-mode", "value"),
)
def update_figure(
    a0, b0, a1, b1, a2, b2,
    t00, t01, t02, t10, t11, t12, t20, t21, t22,
    max_seq_len, use_stationary_list, init_a, init_b, color_mode,
):
    """Rebuild the HMM and update all three plots."""
    s_params = [a0, b0, a1, b1, a2, b2]
    t_params = [t00, t01, t02, t10, t11, t12, t20, t21, t22]
    use_stationary = "on" in (use_stationary_list or [])
    init_params = [init_a, init_b] if not use_stationary else None

    # Build HMM
    try:
        hmm = build_hmm(s_params, t_params, use_stationary, init_params)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error building HMM: {e}", showarrow=False, font=dict(size=14, color="red"))
        return fig, f"Error: {e}"

    # Enumerate beliefs
    try:
        beliefs, obs_probs, log_obs_probs, min_depths, counts, sequences = enumerate_beliefs_and_obs(hmm, max_seq_len)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error enumerating beliefs: {e}", showarrow=False, font=dict(size=14, color="red"))
        return fig, f"Error: {e}"

    n = len(beliefs)

    # Project beliefs
    bx, by = project_beliefs_to_2d(beliefs)

    # Colours
    colors = compute_colors(beliefs, obs_probs, log_obs_probs, min_depths, color_mode)
    use_colorscale = color_mode == "depth"

    # Hover texts per subplot
    hover_belief = [
        f"seq: {_format_seq(sequences[i])}<br>"
        f"b=({beliefs[i, 0]:.4f}, {beliefs[i, 1]:.4f}, {beliefs[i, 2]:.4f})<br>"
        f"depth={min_depths[i]}, overlaps={counts[i]}"
        for i in range(n)
    ]
    hover_obs = [
        f"seq: {_format_seq(sequences[i])}<br>"
        f"P(obs=0)={obs_probs[i, 0]:.4f}<br>"
        f"P(obs=1)={obs_probs[i, 1]:.4f}<br>"
        f"depth={min_depths[i]}, overlaps={counts[i]}"
        for i in range(n)
    ]
    hover_log = [
        f"seq: {_format_seq(sequences[i])}<br>"
        f"log P(obs=0)={log_obs_probs[i, 0]:.4f}<br>"
        f"log P(obs=1)={log_obs_probs[i, 1]:.4f}<br>"
        f"depth={min_depths[i]}, overlaps={counts[i]}"
        for i in range(n)
    ]

    # Build figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Belief states", "Observation probabilities", "Log observation probabilities"],
        horizontal_spacing=0.06,
    )

    marker_kwargs: dict
    if use_colorscale:
        marker_kwargs = dict(
            size=5, color=colors, colorscale="Viridis", showscale=True,
            colorbar=dict(title="depth", len=0.6),
        )
    else:
        marker_kwargs = dict(size=5, color=colors)

    # Subplot 1: beliefs
    tri_x, tri_y = _simplex_triangle()
    fig.add_trace(go.Scatter(
        x=tri_x, y=tri_y, mode="lines",
        line=dict(color="grey", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    # Vertex labels
    vx, vy = project_beliefs_to_2d(_SIMPLEX_VERTICES)
    for label, x, y in zip(["S0", "S1", "S2"], vx, vy):
        fig.add_annotation(
            x=x, y=y, text=label, showarrow=False,
            font=dict(size=10, color="grey"),
            xshift=0, yshift=12, row=1, col=1,
        )

    fig.add_trace(go.Scatter(
        x=bx, y=by, mode="markers",
        marker=marker_kwargs,
        text=hover_belief, hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)

    # Kernel direction line
    kernel_dir = compute_kernel_direction(np.array(hmm.transition_matrices))
    if kernel_dir is not None:
        endpoints = _kernel_simplex_endpoints(kernel_dir)
        if endpoints is not None:
            p1, p2 = endpoints
            kx1, ky1 = project_beliefs_to_2d(p1.reshape(1, -1))
            kx2, ky2 = project_beliefs_to_2d(p2.reshape(1, -1))
            fig.add_trace(go.Scatter(
                x=[kx1[0], kx2[0]], y=[ky1[0], ky2[0]],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Kernel direction",
                hoverinfo="text",
                text=["kernel of belief→obs map"] * 2,
                showlegend=True,
            ), row=1, col=1)

    # Subplot 2: obs probs with reference line y = 1 - x
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 0], mode="lines",
        line=dict(color="grey", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=obs_probs[:, 0], y=obs_probs[:, 1], mode="markers",
        marker=marker_kwargs,
        text=hover_obs, hoverinfo="text",
        showlegend=False,
    ), row=1, col=2)

    # Subplot 3: log obs probs
    fig.add_trace(go.Scatter(
        x=log_obs_probs[:, 0], y=log_obs_probs[:, 1], mode="markers",
        marker=marker_kwargs,
        text=hover_log, hoverinfo="text",
        showlegend=False,
    ), row=1, col=3)

    # Axis labels
    fig.update_xaxes(title_text="v1", row=1, col=1)
    fig.update_yaxes(title_text="v2", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="P(obs=0)", row=1, col=2)
    fig.update_yaxes(title_text="P(obs=1)", row=1, col=2)
    fig.update_xaxes(title_text="log P(obs=0)", row=1, col=3)
    fig.update_yaxes(title_text="log P(obs=1)", row=1, col=3)

    fig.update_layout(
        height=550,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    # Info panel
    s = stick_breaking_to_stochastic(s_params)
    t = np.array(t_params).reshape(3, 3)
    init = np.array(hmm._initial_state)
    kernel_info = "Kernel direction: degenerate"
    if kernel_dir is not None:
        kernel_info = f"Kernel direction: [{kernel_dir[0]:.4f}, {kernel_dir[1]:.4f}, {kernel_dir[2]:.4f}]"
    info = (
        f"s matrix:\n{np.array2string(s, precision=3, suppress_small=True)}\n\n"
        f"t matrix:\n{np.array2string(t, precision=3, suppress_small=True)}\n\n"
        f"Initial distribution: [{init[0]:.4f}, {init[1]:.4f}, {init[2]:.4f}]\n"
        f"{kernel_info}\n"
        f"Unique beliefs: {n}\n"
        f"Unique obs probs: {len(set(tuple(round(float(v), 8) for v in row) for row in obs_probs))}\n"
        f"Max sequence length: {max_seq_len}"
    )

    return fig, info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
