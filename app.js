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
      { id: 'x', label: 'x', min: 0.001, max: 0.499, step: 0.001, default: 0.15 },
      { id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.6  },
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
      { id: 'x', label: 'x', min: 0.001, max: 0.499, step: 0.001, default: 0.15 },
      { id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.6  },
      { id: 'p', label: 'p', min: 0.001, max: 0.999, step: 0.001, default: 0.7  },
      { id: 'q', label: 'q', min: 0.001, max: 0.999, step: 0.001, default: 0.3  },
      { id: 'r', label: 'r', min: 0.001, max: 0.999, step: 0.001, default: 0.5  },
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

  strata: {
    label: 'STRATA',
    type: 'simplex',
    states: 3,
    symbols: 2,
    params: [
      { id: 'a',  label: 'a',  min: 0.001, max: 0.999, step: 0.001, default: 0.5 },
      { id: 't0', label: 't₀', min: 0.0,   max: 1.0,   step: 0.001, default: 0.5 },
      { id: 't1', label: 't₁', min: 0.0,   max: 1.0,   step: 0.001, default: 0.5 },
    ],
    description: `
      <strong>STRATA</strong> is a 3-state, 2-symbol HMM. Parameter <em>a</em> sets the
      diagonal persistence; <em>t₀</em> and <em>t₁</em> independently bias the emission
      split inside states 0 and 1.
      <br><br>
      <span class="tag">3 states</span> <span class="tag">2 symbols</span>
      <span class="tag">HMM</span> <span class="tag">2-simplex</span>
    `,
  },

  arch: {
    label: 'ARCH',
    type: 'pca',
    states: 4,
    symbols: 3,
    params: [
      { id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.5 },
    ],
    description: `
      <strong>ARCH</strong> is a 4-state, 3-symbol HMM. Belief states live on the
      3-simplex (tetrahedron); the visualization projects them via PCA.
      <br><br>
      A single <em>a</em> controls the chain's persistence vs. mixing. The off-diagonal
      mass is split uniformly with rate <em>(1 − a)/3</em>.
      <br><br>
      <span class="tag">4 states</span> <span class="tag">3 symbols</span>
      <span class="tag">HMM</span> <span class="tag">PCA</span>
    `,
  },

  wing: {
    label: 'WING',
    type: 'simplex',
    states: 3,
    symbols: 2,
    params: [
      { id: 'x', label: 'x', min: 0.001, max: 0.999, step: 0.001, default: 0.99 },
      { id: 'y', label: 'y', min: 0.001, max: 0.999, step: 0.001, default: 0.4  },
    ],
    description: `
      <strong>WING</strong> is a 3-state, 2-symbol HMM. Parameter <em>x</em> controls the
      diagonal persistence of the transition tensor; <em>y</em> shifts probability mass
      between the two emissions inside the middle state.
      <br><br>
      Belief states form a wing-like fan emanating from one corner of the 2-simplex,
      with finer striations as <em>x</em> grows.
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
const BG       = '#08090c';
const PLOT_BG  = '#0f1014';
const GRID_COL = '#1a1b20';
const TEXT_COL = '#66635b';
const TICK_COL = '#8a8478';

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
    obsProbPlot:   $('obs-prob-plot'),
    logObsProbPlot:$('log-obs-prob-plot'),
    paramGridPlot: $('param-grid-plot'),
    paramGridWrap: $('param-grid-wrapper'),
    gridScaleTog:  $('grid-scale-toggle'),
    structuralSec: $('structural-section'),
    metricsGrid:   $('metrics-grid'),
    metricsNote:   $('metrics-note'),
    mMu:           $('metric-mu'),
    mLambda2:      $('metric-lambda2'),
    mPi:           $('metric-pi'),
    mHA:           $('metric-HA'),
    mHB:           $('metric-HB'),
    mhA:           $('metric-hA'),
    mhB:           $('metric-hB'),
    mhTot:         $('metric-htot'),
    mMixTime:      $('metric-mixtime'),
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

  setupGridScaleToggle();
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

  const decsFor = (p) => p.step < 0.01 ? 3 : (p.step < 0.1 ? 2 : (p.step < 10 ? 1 : 0));

  DOM.paramSec.style.display = 'block';
  DOM.paramList.innerHTML = cfg.params.map(p => {
    const decs = decsFor(p);
    return `
      <div class="param-row">
        <div class="param-header">
          <label class="param-label">${p.label}</label>
          <input type="number" class="param-val param-num" id="val-${p.id}"
                 min="${p.min}" max="${p.max}" step="${p.step}"
                 value="${p.default.toFixed(decs)}" />
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
    const sl   = $(`slider-${p.id}`);
    const vEl  = $(`val-${p.id}`);
    const decs = decsFor(p);
    const clamp = (v) => Math.min(p.max, Math.max(p.min, v));

    sl.addEventListener('input', () => {
      vEl.value = parseFloat(sl.value).toFixed(decs);
      scheduleAutoCompute();
    });
    vEl.addEventListener('input', () => {
      const v = parseFloat(vEl.value);
      if (Number.isFinite(v)) {
        sl.value = clamp(v);
        scheduleAutoCompute();
      }
    });
    vEl.addEventListener('change', () => {
      const v = parseFloat(vEl.value);
      if (!Number.isFinite(v)) {
        vEl.value = parseFloat(sl.value).toFixed(decs);
        return;
      }
      const c = clamp(v);
      sl.value = c;
      vEl.value = c.toFixed(decs);
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
function compute() {
  if (isComputing) return;
  clearTimeout(debounceTimer);

  isComputing = true;
  DOM.computeBtn.disabled = true;
  DOM.emptyState.classList.add('hidden');
  // Allow loading overlay to paint before blocking computation
  clearTimeout(loadingTimer);
  loadingTimer = setTimeout(() => DOM.loadingOv.classList.remove('hidden'), 50);

  const proc   = currentProcess;
  const params = getParams();

  // Use setTimeout(0) so the loading overlay can paint before the sync computation runs
  setTimeout(() => {
    const t0 = performance.now();
    try {
      const modeParams = currentMode === 'sample'
        ? { batchSize: parseInt(DOM.batchSizeSl.value), seqLen: parseInt(DOM.seqLenSl.value), seed: seedCounter++ }
        : { maxSeqLen: parseInt(DOM.depthSl.value) };

      const data    = computeResult(proc, params, currentMode, modeParams);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

      if (proc !== currentProcess) return;

      DOM.statNodes.textContent = data.n?.toLocaleString() ?? '\u2014';
      DOM.statTime.textContent  = `${elapsed}s`;

      renderVisualization(proc, data, params);
      renderStructuralPanel(data.metrics, proc);
      renderParamGrid(proc, params);

    } catch (err) {
      console.error('Compute error:', err);
      showError(err.message);
    } finally {
      isComputing = false;
      clearTimeout(loadingTimer);
      DOM.loadingOv.classList.add('hidden');
      DOM.computeBtn.disabled = false;
    }
  }, 0);
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

  // Show observation probability plots
  DOM.obsProbPlot.classList.remove('hidden');
  DOM.logObsProbPlot.classList.remove('hidden');
  plotObsProbs(data, 'obs-prob-plot');
  plotLogObsProbs(data, 'log-obs-prob-plot');
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
    line: { color: '#1e2029', width: 2 },
    showlegend: false, hoverinfo: 'skip',
  };

  const vtxMarkers = {
    x: V.map(v => v[0]), y: V.map(v => v[1]),
    mode: 'markers+text', type: 'scatter',
    text: ['S\u2080', 'S\u2081', 'S\u2082'],
    textposition: ['bottom left', 'bottom right', 'top center'],
    textfont: { size: 14, color: '#66635b', family: 'Fira Code, monospace' },
    marker: { color: '#66635b', size: 9 },
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
    title: { text: title, font: { size: 12, color: '#8a8478' }, x: 0.03, xanchor: 'left' },
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
        bgcolor: 'rgba(0,0,0,0)', bordercolor: '#1e2029', tickformat: '.2f',
      },
    },
    hovertemplate:
      'P(obs=0) = %{marker.color:.5f}<br>PC1 = %{x:.4f}<br>PC2 = %{y:.4f}<extra></extra>',
    name: `${n.toLocaleString()} points`,
    showlegend: true,
  };

  const layout = {
    title: { text: title, font: { size: 12, color: '#8a8478' }, x: 0.03, xanchor: 'left' },
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
  const n = Math.min(sorted.length, RUG_MAX_POINTS);
  const xs = sorted.length <= RUG_MAX_POINTS ? Array.from(sorted) : (() => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = sorted[Math.round(i * (sorted.length - 1) / (n - 1))];
    return Array.from(out);
  })();

  const rug = {
    x: xs, y: new Float64Array(xs.length),
    mode: 'markers', type: 'scatter',
    marker: {
      color: xs, colorscale: 'RdBu', reversescale: true,
      size: 3, symbol: 'line-ns',
      line: { width: 1.5, color: xs, colorscale: 'RdBu', reversescale: true },
    },
    hovertemplate: 'P(obs=0) = %{x:.6f}<extra></extra>',
    showlegend: false,
  };

  const layout = {
    title: {
      text: 'P(next obs = 0)  \u00b7  Cantor set structure',
      font: { size: 11, color: '#8a8478' }, x: 0.02, xanchor: 'left',
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

// ── Observation probability rug plots ──────────────────────────────────────────
const SYMBOL_COLORS = ['#c9944a', '#5b9a8b', '#8bba7f', '#d4685a', '#ddb070'];
const RUG_MAX_POINTS = 1500;

function subsampleSorted(arr) {
  const sorted = Float64Array.from(arr).sort();
  if (sorted.length <= RUG_MAX_POINTS) return Array.from(sorted);
  // Uniform subsample preserving min/max
  const out = new Float64Array(RUG_MAX_POINTS);
  for (let i = 0; i < RUG_MAX_POINTS; i++) {
    out[i] = sorted[Math.round(i * (sorted.length - 1) / (RUG_MAX_POINTS - 1))];
  }
  return Array.from(out);
}

function plotObsProbs(data, containerId) {
  const { obsProbs, numSymbols } = data;
  const traces = [];

  for (let s = 0; s < numSymbols; s++) {
    const xs = subsampleSorted(obsProbs[s]);
    traces.push({
      x: xs,
      y: new Array(xs.length).fill(s),
      mode: 'markers', type: 'scatter',
      marker: {
        color: SYMBOL_COLORS[s], size: 3, symbol: 'line-ns', opacity: 0.7,
        line: { width: 1.5, color: SYMBOL_COLORS[s] },
      },
      hovertemplate: `P(obs=${s}) = %{x:.6f}<extra>Symbol ${s}</extra>`,
      showlegend: true,
      name: `sym ${s}`,
    });
  }

  const layout = {
    title: {
      text: 'Observation probabilities  \u00b7  P(next obs = s)',
      font: { size: 11, color: '#8a8478' }, x: 0.02, xanchor: 'left',
    },
    xaxis: {
      range: [-0.02, 1.02], color: TEXT_COL, gridcolor: GRID_COL,
      zeroline: false, tickfont: { size: 9, color: TICK_COL },
      title: { text: 'P(obs = s)', font: { size: 10, color: TEXT_COL } },
    },
    yaxis: { visible: false, range: [-0.8, numSymbols - 0.2] },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 42, l: 10, r: 10 },
    legend: {
      font: { color: TEXT_COL, size: 10 }, bgcolor: 'rgba(0,0,0,0)',
      orientation: 'h', x: 0.75, y: 1.0,
    },
    autosize: true,
  };

  Plotly.react(containerId, traces, layout, plotConfig());
}

function plotLogObsProbs(data, containerId) {
  const { obsProbs, numSymbols } = data;
  const traces = [];
  let minLog = 0;

  for (let s = 0; s < numSymbols; s++) {
    const logVals = new Float64Array(obsProbs[s].length);
    for (let i = 0; i < obsProbs[s].length; i++) {
      logVals[i] = obsProbs[s][i] > 0 ? Math.log(obsProbs[s][i]) : -20;
      if (logVals[i] < minLog) minLog = logVals[i];
    }
    const xs = subsampleSorted(logVals);
    traces.push({
      x: xs,
      y: new Array(xs.length).fill(s),
      mode: 'markers', type: 'scatter',
      marker: {
        color: SYMBOL_COLORS[s], size: 3, symbol: 'line-ns', opacity: 0.7,
        line: { width: 1.5, color: SYMBOL_COLORS[s] },
      },
      hovertemplate: `log P(obs=${s}) = %{x:.4f}<extra>Symbol ${s}</extra>`,
      showlegend: true,
      name: `sym ${s}`,
    });
  }

  const layout = {
    title: {
      text: 'Log observation probabilities  \u00b7  log P(next obs = s)',
      font: { size: 11, color: '#8a8478' }, x: 0.02, xanchor: 'left',
    },
    xaxis: {
      range: [Math.min(minLog * 1.1, -0.5), 0.1], color: TEXT_COL, gridcolor: GRID_COL,
      zeroline: false, tickfont: { size: 9, color: TICK_COL },
      title: { text: 'log P(obs = s)', font: { size: 10, color: TEXT_COL } },
    },
    yaxis: { visible: false, range: [-0.8, numSymbols - 0.2] },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 42, l: 10, r: 10 },
    legend: {
      font: { color: TEXT_COL, size: 10 }, bgcolor: 'rgba(0,0,0,0)',
      orientation: 'h', x: 0.75, y: 1.0,
    },
    autosize: true,
  };

  Plotly.react(containerId, traces, layout, plotConfig());
}

function plotConfig() {
  return {
    responsive: true, displayModeBar: true, displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d'],
  };
}

// ══════════════════════════════════════════════════════════════════════
//  STRUCTURAL METRICS (μ, entropies, h_total)
// ══════════════════════════════════════════════════════════════════════

function fmtFixed(v, d = 3) {
  if (v === Infinity) return '\u221E';
  if (!Number.isFinite(v)) return '\u2014';
  return v.toFixed(d);
}

function fmtSigned(v, d = 3) {
  if (!Number.isFinite(v)) return '\u2014';
  // Use Unicode minus for negatives.
  const s = v.toFixed(d);
  return s.startsWith('-') ? '\u2212' + s.slice(1) : s;
}

function fmtComplex(z, d = 3) {
  if (Math.abs(z.im) < 1e-10) return fmtSigned(z.re, d);
  const sign = z.im >= 0 ? '+' : '\u2212';
  return `${fmtSigned(z.re, d)} ${sign} ${Math.abs(z.im).toFixed(d)}i`;
}

function renderStructuralPanel(metrics, proc) {
  if (!metrics || !metrics.supported) {
    DOM.structuralSec.classList.add('unsupported');
    DOM.metricsGrid.classList.add('hidden');
    DOM.metricsNote.classList.remove('hidden');
    DOM.metricsNote.textContent = proc === 'fanizza'
      ? 'GHMM \u2014 stationary-\u03c0 entropies are not defined; mixing rate is the spectral radius of the dynamical operator.'
      : 'Structural metrics not available for this process.';
    return;
  }

  DOM.structuralSec.classList.remove('unsupported');
  DOM.metricsGrid.classList.remove('hidden');
  DOM.metricsNote.classList.add('hidden');

  DOM.mMu.textContent       = fmtFixed(metrics.mu, 4);
  DOM.mLambda2.textContent  = fmtComplex(metrics.lambda2_raw, 3);
  DOM.mPi.textContent       = '(' + metrics.pi.map(p => p.toFixed(3)).join(', ') + ')';
  DOM.mHA.innerHTML         = fmtFixed(metrics.H_A, 4) + ' <span class="metric-unit">nats</span>';
  DOM.mHB.innerHTML         = fmtFixed(metrics.H_B, 4) + ' <span class="metric-unit">nats</span>';
  DOM.mhA.textContent       = fmtFixed(metrics.h_A_tilde, 4);
  DOM.mhB.textContent       = fmtFixed(metrics.h_B_tilde, 4);
  DOM.mhTot.textContent     = fmtFixed(metrics.h_total, 4);
  const mt = metrics.mixingTime;
  DOM.mMixTime.innerHTML    = (mt === Infinity ? '\u221E' : fmtFixed(mt, 2)) + ' <span class="metric-unit">steps</span>';
}

// ══════════════════════════════════════════════════════════════════════
//  PARAMETER-GRID PLOT (μ and h_total over the (a, x) plane)
// ══════════════════════════════════════════════════════════════════════

const MESS3_GRID = (() => {
  const aVals = [];
  const xVals = [];
  const NA = 99, NX = 50;
  for (let i = 0; i < NA; i++) aVals.push(0.001 + i * (0.999 - 0.001) / (NA - 1));
  for (let j = 0; j < NX; j++) xVals.push(0.001 + j * (0.499 - 0.001) / (NX - 1));
  // Build μ and h_total grids: rows indexed by a, columns by x.
  const mu      = Array.from({ length: NA }, () => new Array(NX));
  const hTotal  = Array.from({ length: NA }, () => new Array(NX));
  for (let i = 0; i < NA; i++) {
    for (let j = 0; j < NX; j++) {
      const m = structuralOnly('mess3', { a: aVals[i], x: xVals[j] });
      mu[i][j]     = m.mu;
      hTotal[i][j] = m.h_total;
    }
  }
  return { aVals, xVals, mu, hTotal };
})();

const WING_GRID = (() => {
  const xVals = [];
  const yVals = [];
  const NX = 99, NY = 99;
  for (let j = 0; j < NX; j++) xVals.push(0.001 + j * (0.999 - 0.001) / (NX - 1));
  for (let i = 0; i < NY; i++) yVals.push(0.001 + i * (0.999 - 0.001) / (NY - 1));
  // Rows = y, columns = x.
  const mu     = Array.from({ length: NY }, () => new Array(NX));
  const hTotal = Array.from({ length: NY }, () => new Array(NX));
  for (let i = 0; i < NY; i++) {
    for (let j = 0; j < NX; j++) {
      const m = structuralOnly('wing', { x: xVals[j], y: yVals[i] });
      mu[i][j]     = m ? m.mu : NaN;
      hTotal[i][j] = m ? m.h_total : NaN;
    }
  }
  return { xVals, yVals, mu, hTotal };
})();

const FERN_GRID = (() => {
  const xVals = [];
  const NX = 41;
  for (let j = 0; j < NX; j++) xVals.push(j * (1.0 - 0.0) / (NX - 1));
  const mu = new Array(NX);
  const hTotal = new Array(NX);
  for (let j = 0; j < NX; j++) {
    const m = structuralOnly('fern', { x: xVals[j] });
    mu[j] = m.mu;
    hTotal[j] = m.h_total;
  }
  return { xVals, mu, hTotal };
})();

let entropyScale = 'linear';  // 'linear' | 'log'

function renderParamGrid(proc, params) {
  if (proc === 'mess3') {
    DOM.paramGridWrap.classList.remove('hidden');
    DOM.gridScaleTog.classList.remove('hidden');
    plotMess3Grid(params);
  } else if (proc === 'fern') {
    DOM.paramGridWrap.classList.remove('hidden');
    DOM.gridScaleTog.classList.remove('hidden');
    plotFernGrid(params);
  } else if (proc === 'wing') {
    DOM.paramGridWrap.classList.remove('hidden');
    DOM.gridScaleTog.classList.remove('hidden');
    plotWingGrid(params);
  } else {
    DOM.paramGridWrap.classList.add('hidden');
  }
}

function setupGridScaleToggle() {
  if (!DOM.gridScaleTog) return;
  DOM.gridScaleTog.querySelectorAll('.scale-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const scale = btn.dataset.scale;
      if (scale === entropyScale) return;
      entropyScale = scale;
      DOM.gridScaleTog.querySelectorAll('.scale-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.scale === scale));
      const params = getParams();
      if (currentProcess === 'mess3') plotMess3Grid(params);
      else if (currentProcess === 'fern') plotFernGrid(params);
      else if (currentProcess === 'wing') plotWingGrid(params);
    });
  });
}

function plotMess3Grid(params) {
  const { aVals, xVals, mu, hTotal } = MESS3_GRID;

  const muHeat = {
    z: mu, x: xVals, y: aVals,
    type: 'heatmap',
    colorscale: 'YlOrRd',
    reversescale: false,
    colorbar: {
      title: { text: '\u03bc', font: { color: TICK_COL, size: 11 } },
      thickness: 10, len: 0.85, x: 0.45, xanchor: 'left',
      tickfont: { color: TICK_COL, size: 9 },
      bgcolor: 'rgba(0,0,0,0)',
      tickformat: '.2f',
    },
    hovertemplate: 'a=%{y:.3f}, x=%{x:.3f}<br>\u03bc=%{z:.3f}<extra></extra>',
    xaxis: 'x', yaxis: 'y',
  };
  const hHeat = {
    z: hTotal, x: xVals, y: aVals,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: {
      title: { text: 'h\u2009total', font: { color: TICK_COL, size: 11 } },
      thickness: 10, len: 0.85, x: 1.0, xanchor: 'left',
      tickfont: { color: TICK_COL, size: 9 },
      bgcolor: 'rgba(0,0,0,0)',
      tickformat: '.2f',
    },
    hovertemplate: 'a=%{y:.3f}, x=%{x:.3f}<br>h_total=%{z:.3f}<extra></extra>',
    xaxis: 'x2', yaxis: 'y2',
  };

  const a0 = params.a ?? 0.6;
  const x0 = params.x ?? 0.15;
  const markerColor = '#ddb070';
  const markerLine  = { color: '#08090c', width: 2 };
  const markerL = {
    x: [x0], y: [a0], type: 'scatter', mode: 'markers',
    marker: { color: markerColor, size: 11, symbol: 'circle', line: markerLine },
    showlegend: false, hoverinfo: 'skip',
    xaxis: 'x', yaxis: 'y',
  };
  const markerR = {
    x: [x0], y: [a0], type: 'scatter', mode: 'markers',
    marker: { color: markerColor, size: 11, symbol: 'circle', line: markerLine },
    showlegend: false, hoverinfo: 'skip',
    xaxis: 'x2', yaxis: 'y2',
  };

  const isLog = entropyScale === 'log';
  const axType = isLog ? 'log' : 'linear';
  const axisStyle = {
    color: TEXT_COL, gridcolor: GRID_COL, zeroline: false,
    tickfont: { size: 9, color: TICK_COL },
  };

  const layout = {
    title: {
      text: `MESS3 structural sweep  \u00b7  \u03bc(a,x)  &  h_total(a,x)  \u00b7  marker = current${isLog ? '  \u00b7  log axes' : ''}`,
      font: { size: 11, color: '#8a8478' }, x: 0.02, xanchor: 'left',
    },
    grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis:  { ...axisStyle, type: axType, domain: [0.0, 0.42], title: { text: 'x', font: { size: 11, color: TEXT_COL } } },
    yaxis:  { ...axisStyle, type: axType, domain: [0.0, 1.0], title: { text: 'a', font: { size: 11, color: TEXT_COL } } },
    xaxis2: { ...axisStyle, type: axType, domain: [0.55, 0.97], title: { text: 'x', font: { size: 11, color: TEXT_COL } } },
    yaxis2: { ...axisStyle, type: axType, domain: [0.0, 1.0], anchor: 'x2', title: { text: 'a', font: { size: 11, color: TEXT_COL } } },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 42, l: 42, r: 10 },
    showlegend: false,
    autosize: true,
  };

  Plotly.react('param-grid-plot', [muHeat, hHeat, markerL, markerR], layout, plotConfig());
}

function plotWingGrid(params) {
  const { xVals, yVals, mu, hTotal } = WING_GRID;

  const muHeat = {
    z: mu, x: xVals, y: yVals,
    type: 'heatmap',
    colorscale: 'YlOrRd',
    colorbar: {
      title: { text: 'μ', font: { color: TICK_COL, size: 11 } },
      thickness: 10, len: 0.85, x: 0.45, xanchor: 'left',
      tickfont: { color: TICK_COL, size: 9 },
      bgcolor: 'rgba(0,0,0,0)',
      tickformat: '.2f',
    },
    hovertemplate: 'y=%{y:.3f}, x=%{x:.3f}<br>μ=%{z:.3f}<extra></extra>',
    xaxis: 'x', yaxis: 'y',
  };
  const hHeat = {
    z: hTotal, x: xVals, y: yVals,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: {
      title: { text: 'h total', font: { color: TICK_COL, size: 11 } },
      thickness: 10, len: 0.85, x: 1.0, xanchor: 'left',
      tickfont: { color: TICK_COL, size: 9 },
      bgcolor: 'rgba(0,0,0,0)',
      tickformat: '.2f',
    },
    hovertemplate: 'y=%{y:.3f}, x=%{x:.3f}<br>h_total=%{z:.3f}<extra></extra>',
    xaxis: 'x2', yaxis: 'y2',
  };

  const x0 = params.x ?? 0.99;
  const y0 = params.y ?? 0.4;
  const markerColor = '#ddb070';
  const markerLine  = { color: '#08090c', width: 2 };
  const markerL = {
    x: [x0], y: [y0], type: 'scatter', mode: 'markers',
    marker: { color: markerColor, size: 11, symbol: 'circle', line: markerLine },
    showlegend: false, hoverinfo: 'skip', xaxis: 'x', yaxis: 'y',
  };
  const markerR = {
    x: [x0], y: [y0], type: 'scatter', mode: 'markers',
    marker: { color: markerColor, size: 11, symbol: 'circle', line: markerLine },
    showlegend: false, hoverinfo: 'skip', xaxis: 'x2', yaxis: 'y2',
  };

  const isLog = entropyScale === 'log';
  const axType = isLog ? 'log' : 'linear';
  const axisStyle = {
    color: TEXT_COL, gridcolor: GRID_COL, zeroline: false,
    tickfont: { size: 9, color: TICK_COL },
  };

  const layout = {
    title: {
      text: `WING structural sweep  ·  μ(x,y)  &  h_total(x,y)  ·  marker = current${isLog ? '  ·  log axes' : ''}`,
      font: { size: 11, color: '#8a8478' }, x: 0.02, xanchor: 'left',
    },
    grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis:  { ...axisStyle, type: axType, domain: [0.0, 0.42], title: { text: 'x', font: { size: 11, color: TEXT_COL } } },
    yaxis:  { ...axisStyle, type: axType, domain: [0.0, 1.0], title: { text: 'y', font: { size: 11, color: TEXT_COL } } },
    xaxis2: { ...axisStyle, type: axType, domain: [0.55, 0.97], title: { text: 'x', font: { size: 11, color: TEXT_COL } } },
    yaxis2: { ...axisStyle, type: axType, domain: [0.0, 1.0], anchor: 'x2', title: { text: 'y', font: { size: 11, color: TEXT_COL } } },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 42, l: 42, r: 10 },
    showlegend: false,
    autosize: true,
  };

  Plotly.react('param-grid-plot', [muHeat, hHeat, markerL, markerR], layout, plotConfig());
}

function plotFernGrid(params) {
  const { xVals, mu, hTotal } = FERN_GRID;
  const x0 = params.x ?? 0.5;

  // Find current values via interpolation on the grid (cheap: re-evaluate)
  const cur = structuralOnly('fern', { x: x0 });

  const muLine = {
    x: xVals, y: mu, type: 'scatter', mode: 'lines',
    line: { color: '#c9944a', width: 2, dash: 'dash' },
    name: '\u03bc',
    hovertemplate: 'x=%{x:.3f}<br>\u03bc=%{y:.4f}<extra></extra>',
  };
  const hLine = {
    x: xVals, y: hTotal, type: 'scatter', mode: 'lines',
    line: { color: '#5b9a8b', width: 2 },
    name: 'h_total',
    hovertemplate: 'x=%{x:.3f}<br>h_total=%{y:.4f}<extra></extra>',
  };
  const muMark = {
    x: [x0], y: [cur.mu], type: 'scatter', mode: 'markers',
    marker: { color: '#ddb070', size: 10, line: { color: '#08090c', width: 2 } },
    showlegend: false, hoverinfo: 'skip',
  };
  const hMark = {
    x: [x0], y: [cur.h_total], type: 'scatter', mode: 'markers',
    marker: { color: '#8bba7f', size: 10, line: { color: '#08090c', width: 2 } },
    showlegend: false, hoverinfo: 'skip',
  };

  const layout = {
    title: {
      text: 'FERN structural sweep  \u00b7  \u03bc is x-independent; only h_total varies',
      font: { size: 11, color: '#8a8478' }, x: 0.02, xanchor: 'left',
    },
    xaxis: {
      title: { text: 'x', font: { size: 11, color: TEXT_COL } },
      color: TEXT_COL, gridcolor: GRID_COL, zeroline: false,
      tickfont: { size: 10, color: TICK_COL },
      type: entropyScale === 'log' ? 'log' : 'linear',
      ...(entropyScale === 'log' ? {} : { range: [-0.02, 1.02] }),
    },
    yaxis: {
      title: { text: 'value', font: { size: 11, color: TEXT_COL } },
      color: TEXT_COL, gridcolor: GRID_COL, zeroline: false,
      tickfont: { size: 10, color: TICK_COL },
    },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 48, l: 50, r: 16 },
    legend: {
      font: { color: TEXT_COL, size: 10 }, bgcolor: 'rgba(0,0,0,0)',
      orientation: 'h', x: 0.75, y: 1.0,
    },
    autosize: true,
  };

  Plotly.react('param-grid-plot', [muLine, hLine, muMark, hMark], layout, plotConfig());
}
