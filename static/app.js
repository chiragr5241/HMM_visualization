/* ═══════════════════════════════════════════════════════════════════
   Belief State Geometry — frontend logic
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ── Process catalogue ──────────────────────────────────────────────────────────
const PROCESSES = {
  mess3: {
    label: 'MESS3',
    type: 'simplex',
    states: 3,
    symbols: 3,
    params: [
      { id: 'x', label: 'x', min: 0.01, max: 0.49, step: 0.01, default: 0.15 },
      { id: 'a', label: 'a', min: 0.1,  max: 0.99, step: 0.01, default: 0.6  },
    ],
    description: `
      <strong>MESS3</strong> is a 3-state, 3-symbol HMM with <strong>\u2124\u2083 symmetry</strong>.
      The MSP is a self-similar fractal on the 2-simplex.
      <br><br>
      Higher <em>a</em> \u2192 belief states cluster near the vertices (strong state memory).
      Lower <em>a</em> \u2192 states spread toward the centroid (more mixing).
      Parameter <em>x</em> controls the cross-state transition spread.
      <br><br>
      <span class="tag">3 states</span> <span class="tag">3 symbols</span>
      <span class="tag">HMM</span> <span class="tag">2-simplex</span>
    `,
  },

  mess3_2: {
    label: 'MESS3-2',
    type: 'simplex',
    states: 3,
    symbols: 2,
    params: [
      { id: 'x', label: 'x', min: 0.01, max: 0.49, step: 0.01, default: 0.15 },
      { id: 'a', label: 'a', min: 0.1,  max: 0.99, step: 0.01, default: 0.6  },
      { id: 'p', label: 'p', min: 0.01, max: 0.99, step: 0.01, default: 0.7  },
      { id: 'q', label: 'q', min: 0.01, max: 0.99, step: 0.01, default: 0.3  },
      { id: 'r', label: 'r', min: 0.01, max: 0.99, step: 0.01, default: 0.5  },
    ],
    description: `
      <strong>MESS3-2</strong> collapses MESS3's 3 emissions into 2 via weighted mixing.
      The 3-dimensional hidden state space is unchanged, so belief states
      still live on the 2-simplex \u2014 but the geometry differs.
      <br><br>
      Parameters <em>p</em>, <em>q</em>, <em>r</em> control the per-state mixing weights
      for the two output symbols.
      <br><br>
      <span class="tag">3 states</span> <span class="tag">2 symbols</span>
      <span class="tag">HMM</span> <span class="tag">2-simplex</span>
    `,
  },

  river: {
    label: 'RIVER',
    type: 'simplex',
    states: 3,
    symbols: 2,
    params: [],
    description: `
      <strong>RIVER</strong> is a fixed 3-state, 2-symbol HMM with no free parameters.
      Its asymmetric sparse transitions produce a distinctive <strong>irregular fractal</strong>
      on the 2-simplex \u2014 very different from MESS3's \u2124\u2083-symmetric geometry.
      <br><br>
      Zero entries in the transition matrices are handled via
      log-space arithmetic (log 0 = \u2212\u221e, logsumexp-safe).
      <br><br>
      <span class="tag">3 states</span> <span class="tag">2 symbols</span>
      <span class="tag">HMM</span> <span class="tag">fixed</span>
    `,
  },

  leopard: {
    label: 'LEOPARD',
    type: 'simplex',
    states: 3,
    symbols: 2,
    params: [
      { id: 'x', label: 'x', min: 0.0, max: 1.0, step: 0.01, default: 0.5 },
    ],
    description: `
      <strong>LEOPARD</strong> is a 3-state, 2-symbol HMM from the
      <a href="https://github.com/Astera-org/simplexity/blob/xavier/processes" target="_blank">xavier/processes</a> branch.
      <br><br>
      The single parameter <em>x</em> shifts probability mass between two cyclic emission channels,
      continuously morphing the MSP pattern on the 2-simplex.
      <br><br>
      <span class="tag">3 states</span> <span class="tag">2 symbols</span>
      <span class="tag">HMM</span> <span class="tag">2-simplex</span>
    `,
  },

  fern: {
    label: 'FERN',
    type: 'simplex',
    states: 3,
    symbols: 2,
    params: [
      { id: 'x', label: 'x', min: 0.0, max: 1.0, step: 0.01, default: 0.5 },
    ],
    description: `
      <strong>FERN</strong> is a 3-state, 2-symbol HMM from the
      <a href="https://github.com/Astera-org/simplexity/blob/xavier/processes" target="_blank">xavier/processes</a> branch.
      <br><br>
      The parameter <em>x</em> controls the balance of transition mass in the third state,
      producing intricate fern-like fractal patterns on the 2-simplex.
      <br><br>
      <span class="tag">3 states</span> <span class="tag">2 symbols</span>
      <span class="tag">HMM</span> <span class="tag">2-simplex</span>
    `,
  },

  fanizza: {
    label: 'FANIZZA',
    type: 'pca',
    states: 4,
    symbols: 2,
    params: [
      { id: 'alpha', label: '\u03b1  (radians)', min: 10,   max: 4000, step: 10,   default: 2000 },
      { id: 'lamb',  label: '\u03bb',            min: 0.05, max: 0.99, step: 0.01, default: 0.49 },
    ],
    description: `
      <strong>FANIZZA</strong> is a 4-state <strong>Generalized HMM</strong>.
      Belief states are <em>not</em> probability vectors \u2014 they live in an affine subspace
      normalised by the principal eigenvector of the state-transition operator.
      <br><br>
      When <em>\u03b1 / 2\u03c0</em> is irrational and <em>\u03bb &lt; 1</em>,
      the MSP traces a <strong>Cantor set</strong> \u2014 a fractal of measure zero.
      The 1D rug plot below shows this gap structure in P(obs=0).
      <br><br>
      <span class="tag">4 states</span> <span class="tag">2 symbols</span>
      <span class="tag">GHMM</span> <span class="tag">Cantor set</span>
    `,
  },
};

// ── Plotly theme constants ─────────────────────────────────────────────────────
const BG       = '#0b0d18';
const PLOT_BG  = '#10121f';
const GRID_COL = '#1a1d30';
const TEXT_COL = '#676b8c';
const TICK_COL = '#9097b8';

// ── Cached DOM refs ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
let DOM;

// ── App state ──────────────────────────────────────────────────────────────────
let currentProcess = 'mess3';
let currentMode    = 'sample';      // 'sample' or 'enumerate'
let isComputing    = false;
let debounceTimer  = null;
let loadingTimer   = null;
let seedCounter    = 0;

// ── Boot ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  DOM = {
    paramList:     $('param-list'),
    paramSec:      $('param-section'),
    desc:          $('description'),
    statNodes:     $('stat-nodes'),
    statTime:      $('stat-time'),
    statStates:    $('stat-states'),
    statSymbols:   $('stat-symbols'),
    computeBtn:    $('compute-btn'),
    loadingOv:     $('loading-overlay'),
    emptyState:    $('empty-state'),
    cantorPlot:    $('cantor-plot'),
    mainPlot:      $('main-plot'),
    // Sample mode controls
    batchSizeSl:   $('batch-size'),
    batchSizeVal:  $('batch-size-val'),
    seqLenSl:      $('seq-len'),
    seqLenVal:     $('seq-len-val'),
    pointEst:      $('point-estimate'),
    sampleSection: $('sample-section'),
    // Enumerate mode controls
    depthSl:       $('max-seq-len'),
    depthVal:      $('max-seq-len-val'),
    nodeEst:       $('node-estimate'),
    enumSection:   $('enum-section'),
    // Mode toggle
    modeSample:    $('mode-sample'),
    modeEnum:      $('mode-enum'),
  };

  // Wire process buttons
  document.querySelectorAll('.proc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.proc !== currentProcess) selectProcess(btn.dataset.proc);
    });
  });

  // Wire mode toggle
  DOM.modeSample.addEventListener('click', () => setMode('sample'));
  DOM.modeEnum.addEventListener('click', () => setMode('enumerate'));

  // Wire compute button
  DOM.computeBtn.addEventListener('click', compute);

  // Wire sample controls
  DOM.batchSizeSl.addEventListener('input', () => {
    DOM.batchSizeVal.textContent = DOM.batchSizeSl.value;
    updatePointEstimate();
    scheduleAutoCompute();
  });
  DOM.seqLenSl.addEventListener('input', () => {
    DOM.seqLenVal.textContent = DOM.seqLenSl.value;
    updatePointEstimate();
    scheduleAutoCompute();
  });

  // Wire enumerate depth slider
  DOM.depthSl.addEventListener('input', () => {
    DOM.depthVal.textContent = DOM.depthSl.value;
    updateNodeEstimate();
  });

  selectProcess('mess3', true);
});

// ── Mode switching ─────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  DOM.modeSample.classList.toggle('active', mode === 'sample');
  DOM.modeEnum.classList.toggle('active', mode === 'enumerate');
  DOM.sampleSection.classList.toggle('hidden', mode !== 'sample');
  DOM.enumSection.classList.toggle('hidden', mode !== 'enumerate');
  // Update compute button label
  DOM.computeBtn.querySelector('.btn-label').textContent =
    mode === 'sample' ? 'Sample' : 'Enumerate MSP';
}

// ── Process selection ──────────────────────────────────────────────────────────
function selectProcess(proc, autoCompute = true) {
  currentProcess = proc;
  document.querySelectorAll('.proc-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.proc === proc)
  );

  const cfg = PROCESSES[proc];
  renderParams(cfg);

  DOM.desc.innerHTML = cfg.description;
  DOM.statStates.textContent  = cfg.states;
  DOM.statSymbols.textContent = cfg.symbols;
  DOM.statNodes.textContent = '\u2014';
  DOM.statTime.textContent  = '\u2014';

  updatePointEstimate();
  updateNodeEstimate();

  if (autoCompute) compute();
}

// ── Parameter panel ────────────────────────────────────────────────────────────
function renderParams(cfg) {
  if (!cfg.params.length) {
    DOM.paramList.innerHTML = '<div class="no-params">No parameters \u2014 fixed process</div>';
    DOM.paramSec.style.display = 'block';
    return;
  }

  DOM.paramSec.style.display = 'block';
  DOM.paramList.innerHTML = cfg.params.map(p => {
    const decs = p.step < 0.1 ? 2 : (p.step < 10 ? 1 : 0);
    return `
      <div class="param-row">
        <div class="param-header">
          <label class="param-label">${p.label}</label>
          <span class="param-val" id="val-${p.id}">${p.default.toFixed(decs)}</span>
        </div>
        <div class="slider-row">
          <span class="slider-bound">${p.min}</span>
          <input type="range" class="slider" id="slider-${p.id}"
                 min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}" />
          <span class="slider-bound right">${p.max}</span>
        </div>
      </div>
    `;
  }).join('');

  cfg.params.forEach(p => {
    const sl  = $(`slider-${p.id}`);
    const vEl = $(`val-${p.id}`);
    const decs = p.step < 0.1 ? 2 : (p.step < 10 ? 1 : 0);
    sl.addEventListener('input', () => {
      vEl.textContent = parseFloat(sl.value).toFixed(decs);
      scheduleAutoCompute();
    });
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function getParams() {
  return Object.fromEntries(
    PROCESSES[currentProcess].params.map(p => {
      const sl = $(`slider-${p.id}`);
      return [p.id, sl ? parseFloat(sl.value) : p.default];
    })
  );
}

function updatePointEstimate() {
  const batch = parseInt(DOM.batchSizeSl.value);
  const seq   = parseInt(DOM.seqLenSl.value);
  DOM.pointEst.textContent = (batch * (seq + 1)).toLocaleString();
}

function updateNodeEstimate() {
  const depth   = parseInt(DOM.depthSl.value);
  const symbols = PROCESSES[currentProcess].symbols;
  let n = 0;
  for (let d = 0; d <= depth; d++) n += symbols ** d;
  DOM.nodeEst.textContent = n.toLocaleString();
}

function scheduleAutoCompute() {
  clearTimeout(debounceTimer);
  // In sample mode, use short debounce for live updates
  // In enumerate mode, don't auto-compute (user clicks button)
  if (currentMode === 'sample') {
    debounceTimer = setTimeout(compute, 150);
  }
}

// ── Compute ────────────────────────────────────────────────────────────────────
async function compute() {
  if (isComputing) return;
  clearTimeout(debounceTimer);

  isComputing = true;
  DOM.computeBtn.disabled = true;
  DOM.emptyState.classList.add('hidden');
  // Only show loading overlay if computation takes > 200ms
  clearTimeout(loadingTimer);
  loadingTimer = setTimeout(() => DOM.loadingOv.classList.remove('hidden'), 200);

  const proc   = currentProcess;
  const params = getParams();
  const t0     = performance.now();

  const body = { process: proc, params, mode: currentMode };
  if (currentMode === 'sample') {
    body.batch_size = parseInt(DOM.batchSizeSl.value);
    body.seq_len    = parseInt(DOM.seqLenSl.value);
    body.seed       = seedCounter++;
  } else {
    body.max_seq_len = parseInt(DOM.depthSl.value);
  }

  try {
    const resp = await fetch('/api/compute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data    = await resp.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    if (proc !== currentProcess) return;

    DOM.statNodes.textContent = data.n?.toLocaleString() ?? '\u2014';
    DOM.statTime.textContent  = `${elapsed}s`;

    renderVisualization(proc, data, params);

  } catch (err) {
    console.error('Compute error:', err);
    showError(err.message);
  } finally {
    isComputing = false;
    clearTimeout(loadingTimer);
    DOM.loadingOv.classList.add('hidden');
    DOM.computeBtn.disabled = false;
  }
}

// ── Render dispatch ────────────────────────────────────────────────────────────
function renderVisualization(proc, data, params) {
  const title = buildTitle(proc, params, data);

  if (data.type === 'simplex') {
    DOM.cantorPlot.classList.add('hidden');
    plotSimplex(data, 'main-plot', title);
  } else if (data.type === 'pca') {
    DOM.cantorPlot.classList.remove('hidden');
    plotPCA(data, 'main-plot', title);
    plotCantor(data.color_val, 'cantor-plot');
  }
}

function buildTitle(proc, params, data) {
  const cfg = PROCESSES[proc];
  const decs = (p) => p.step < 0.1 ? 2 : (p.step < 10 ? 1 : 0);
  const parts = cfg.params.map(p =>
    `${p.label} = ${(params[p.id] ?? p.default).toFixed(decs(p))}`
  );
  const suffix = parts.length ? `  \u00b7  ${parts.join(',  ')}` : '';
  const modeLabel = data.mode === 'sample' ? 'sampled' : 'enumerated';
  return `${cfg.label}${suffix}  \u00b7  ${data.n.toLocaleString()} ${modeLabel}`;
}

// ── Loading / UI state ─────────────────────────────────────────────────────────
function showLoading(show) {
  DOM.loadingOv.classList.toggle('hidden', !show);
  DOM.computeBtn.disabled = show;
}

function showError(msg) {
  let el = $('error-toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'error-toast';
    el.className = 'error-toast';
    document.querySelector('.main').appendChild(el);
  }
  el.textContent = `Error: ${msg}`;
  el.classList.add('show');
  setTimeout(() => el.remove(), 5000);
}

// ══════════════════════════════════════════════════════════════════════
//  PLOTLY RENDERERS
// ══════════════════════════════════════════════════════════════════════

function traceType(n) { return n > 2000 ? 'scattergl' : 'scatter'; }

// ── Simplex (HMMs) ─────────────────────────────────────────────────────────────
function plotSimplex(data, containerId, title) {
  const sqrt3 = Math.sqrt(3);
  const V = [[0, 0], [1, 0], [0.5, sqrt3 / 2]];
  const n = data.n;

  const colors = new Array(n);
  for (let i = 0; i < n; i++) {
    colors[i] = `rgb(${data.r[i]},${data.g[i]},${data.b[i]})`;
  }

  const markerSize = Math.max(2, Math.min(5, Math.round(8000 / n)));

  const edges = {
    x: [V[0][0], V[1][0], V[2][0], V[0][0]],
    y: [V[0][1], V[1][1], V[2][1], V[0][1]],
    mode: 'lines', type: 'scatter',
    line: { color: '#2a2f50', width: 2 },
    showlegend: false, hoverinfo: 'skip',
  };

  const vtxMarkers = {
    x: V.map(v => v[0]), y: V.map(v => v[1]),
    mode: 'markers+text', type: 'scatter',
    text: ['S\u2080', 'S\u2081', 'S\u2082'],
    textposition: ['bottom left', 'bottom right', 'top center'],
    textfont: { size: 14, color: '#7b7faa', family: 'JetBrains Mono, monospace' },
    marker: { color: '#7b7faa', size: 9 },
    showlegend: false, hoverinfo: 'skip',
  };

  const pts = {
    x: data.x, y: data.y,
    mode: 'markers', type: traceType(n),
    marker: { color: colors, size: markerSize, opacity: 0.88 },
    customdata: data.bs,
    hovertemplate:
      '<b>Belief state</b><br>' +
      'P(S\u2080) = %{customdata[0]:.5f}<br>' +
      'P(S\u2081) = %{customdata[1]:.5f}<br>' +
      'P(S\u2082) = %{customdata[2]:.5f}<extra></extra>',
    name: `${n.toLocaleString()} points`,
    showlegend: true,
  };

  const layout = {
    title: { text: title, font: { size: 12, color: '#9097b8' }, x: 0.03, xanchor: 'left' },
    xaxis: { visible: false, range: [-0.1, 1.1], scaleanchor: 'y', fixedrange: false },
    yaxis: { visible: false, range: [-0.14, sqrt3 / 2 + 0.14], fixedrange: false },
    paper_bgcolor: BG, plot_bgcolor: BG,
    margin: { t: 44, b: 10, l: 10, r: 10 },
    legend: { font: { color: TEXT_COL, size: 11 }, bgcolor: 'rgba(0,0,0,0)', x: 0.01, y: 0.99 },
    autosize: true,
  };

  Plotly.react(containerId, [edges, vtxMarkers, pts], layout, plotConfig());
}

// ── PCA (Fanizza) ──────────────────────────────────────────────────────────────
function plotPCA(data, containerId, title) {
  const [vr1, vr2] = data.var_ratios;
  const n = data.n;

  const pts = {
    x: data.x, y: data.y,
    mode: 'markers', type: traceType(n),
    marker: {
      color: data.color_val, colorscale: 'RdBu', reversescale: true,
      size: Math.max(2, Math.min(5, Math.round(8000 / n))),
      opacity: 0.88,
      colorbar: {
        title: { text: 'P(obs=0)', font: { color: TICK_COL, size: 11 } },
        thickness: 12, len: 0.65,
        tickfont: { color: TICK_COL, size: 10 },
        bgcolor: 'rgba(0,0,0,0)', bordercolor: '#22263d', tickformat: '.2f',
      },
    },
    hovertemplate:
      'P(obs=0) = %{marker.color:.5f}<br>PC1 = %{x:.4f}<br>PC2 = %{y:.4f}<extra></extra>',
    name: `${n.toLocaleString()} points`,
    showlegend: true,
  };

  const layout = {
    title: { text: title, font: { size: 12, color: '#9097b8' }, x: 0.03, xanchor: 'left' },
    xaxis: {
      title: { text: `PC1  (${(vr1 * 100).toFixed(1)}%)`, font: { size: 11, color: TEXT_COL } },
      color: TEXT_COL, gridcolor: GRID_COL, zeroline: false, tickfont: { size: 10, color: TICK_COL },
    },
    yaxis: {
      title: { text: `PC2  (${(vr2 * 100).toFixed(1)}%)`, font: { size: 11, color: TEXT_COL } },
      color: TEXT_COL, gridcolor: GRID_COL, zeroline: false, tickfont: { size: 10, color: TICK_COL },
    },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 44, b: 56, l: 56, r: 20 },
    legend: { font: { color: TEXT_COL, size: 11 }, bgcolor: 'rgba(0,0,0,0)', x: 0.01, y: 0.99 },
    autosize: true,
  };

  Plotly.react(containerId, [pts], layout, plotConfig());
}

// ── Cantor rug (Fanizza) ───────────────────────────────────────────────────────
function plotCantor(cantorVals, containerId) {
  const sorted = Float64Array.from(cantorVals).sort();

  const rug = {
    x: sorted, y: new Float64Array(sorted.length),
    mode: 'markers', type: 'scatter',
    marker: {
      color: Array.from(sorted), colorscale: 'RdBu', reversescale: true,
      size: 3, symbol: 'line-ns',
      line: { width: 1.5, color: Array.from(sorted), colorscale: 'RdBu', reversescale: true },
    },
    hovertemplate: 'P(obs=0) = %{x:.6f}<extra></extra>',
    showlegend: false,
  };

  const layout = {
    title: {
      text: 'P(next obs = 0)  \u00b7  Cantor set structure',
      font: { size: 11, color: '#9097b8' }, x: 0.02, xanchor: 'left',
    },
    xaxis: {
      range: [-0.02, 1.02], color: TEXT_COL, gridcolor: GRID_COL,
      zeroline: false, tickfont: { size: 9, color: TICK_COL },
      title: { text: 'P(obs = 0)', font: { size: 10, color: TEXT_COL } },
    },
    yaxis: { visible: false, range: [-0.8, 0.8] },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 42, l: 10, r: 10 },
    autosize: true,
  };

  Plotly.react(containerId, [rug], layout, plotConfig());
}

function plotConfig() {
  return {
    responsive: true, displayModeBar: true, displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d'],
  };
}
