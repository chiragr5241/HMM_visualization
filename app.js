/* ═══════════════════════════════════════════════════════════════════
   Belief State Geometry — frontend logic
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ── Process catalogue ──────────────────────────────────────────────────────────
// `sweepAxes` picks the default sweep slice; processes with ≥3 parameters let
// the reader choose the two axes.
const PROCESSES = {
  mess3: {
    label: 'MESS3', states: 3, symbols: 3,
    params: [
      { id: 'x', label: 'x', min: 0.001, max: 0.499, step: 0.001, default: 0.15 },
      { id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.6  },
    ],
    description: 'Z₃-symmetric HMM whose beliefs form a self-similar fractal. Higher <em>a</em> pushes them toward the corners.',
  },
  mess3_2: {
    label: 'MESS3-2', states: 3, symbols: 2,
    params: [
      { id: 'x', label: 'x', min: 0.001, max: 0.499, step: 0.001, default: 0.15 },
      { id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.6  },
      { id: 'p', label: 'p', min: 0.001, max: 0.999, step: 0.001, default: 0.7  },
      { id: 'q', label: 'q', min: 0.001, max: 0.999, step: 0.001, default: 0.3  },
      { id: 'r', label: 'r', min: 0.001, max: 0.999, step: 0.001, default: 0.5  },
    ],
    description: 'MESS3 with its three emissions mixed down to two by <em>p, q, r</em>. Same hidden chain, different geometry.',
  },
  river: {
    label: 'RIVER', states: 3, symbols: 2, params: [],
    description: 'A fixed, sparse, asymmetric HMM with an irregular fractal.',
  },
  leopard: {
    label: 'LEOPARD', states: 3, symbols: 2,
    params: [{ id: 'x', label: 'x', min: 0.0, max: 1.0, step: 0.01, default: 0.5 }],
    description: '<em>x</em> shifts mass between two cyclic emission channels.',
  },
  fern: {
    label: 'FERN', states: 3, symbols: 2,
    params: [{ id: 'x', label: 'x', min: 0.0, max: 1.0, step: 0.01, default: 0.5 }],
    description: '<em>x</em> balances transition mass in the third state, giving fern-like fractals.',
  },
  wing: {
    label: 'WING', states: 3, symbols: 2,
    params: [
      { id: 'x', label: 'x', min: 0.001, max: 0.999, step: 0.001, default: 0.99 },
      { id: 'y', label: 'y', min: 0.001, max: 0.999, step: 0.001, default: 0.4  },
    ],
    description: '<em>x</em> sets persistence; <em>y</em> splits the emissions of the middle state.',
  },
  strata: {
    label: 'STRATA', states: 3, symbols: 2,
    params: [
      { id: 'a',  label: 'a',  min: 0.001, max: 0.999, step: 0.001, default: 0.5 },
      { id: 't0', label: 't₀', min: 0.0,   max: 1.0,   step: 0.001, default: 0.5 },
      { id: 't1', label: 't₁', min: 0.0,   max: 1.0,   step: 0.001, default: 0.5 },
    ],
    sweepAxes: ['t0', 'a'],
    description: '<em>a</em> sets persistence; <em>t₀</em> and <em>t₁</em> bias the emissions of states 0 and 1.',
  },
  spiral: {
    label: 'SPIRAL', states: 3, symbols: 2,
    params: [{ id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.5 }],
    description: 'One persistence parameter <em>a</em>; beliefs trace spirals.',
  },
  arch: {
    label: 'ARCH', states: 4, symbols: 3,
    params: [{ id: 'a', label: 'a', min: 0.001, max: 0.999, step: 0.001, default: 0.5 }],
    description: 'Four states, so beliefs fill a tetrahedron (shown by PCA). <em>a</em> trades persistence for mixing.',
  },
  fanizza: {
    label: 'FANIZZA', states: 4, symbols: 2, ghmm: true,
    params: [
      { id: 'alpha', label: 'α  (radians)', min: 10,   max: 4000, step: 10,   default: 2000 },
      { id: 'lamb',  label: 'λ',            min: 0.05, max: 0.99, step: 0.01, default: 0.49 },
    ],
    sweepAxes: ['lamb'],
    description: 'Generalized HMM. With α/2π irrational and λ &lt; 1, P(next = 0) is a Cantor set (see Emissions).',
  },
  bloch: {
    label: 'BLOCH WALK', states: 3, symbols: 4, ghmm: true,
    meta: '1 qubit · 4 symbols · no finite HMM',
    maxDepth: 9,
    sweepScale: 'log',   // β and 1/β give the same s, so log axes show the symmetry
    // 0↔1 and 2↔3 cancel, so depth L reaches 1 + 2(3^L − 1) distinct beliefs.
    nodeEstimate: (depth) => 1 + 2 * (3 ** depth - 1),
    params: [
      { id: 'beta', label: 'β  (α = 1)', min: 0.05, max: 20, step: 0.01, default: Math.sqrt(51) },
    ],
    description: 'One qubit of memory, no finite HMM: beliefs fill a disk, not a triangle. The paper uses β = √51.',
  },
};

// ── Parametrizations from arXiv:2609.17376 (Table 1) ───────────────────────────
// Paper notation → this site: Mess3 (x, α) = (x, a); Arch α = a;
// Wing (α, x) = (x, y); Strata (α, t₀, t₁) = (a, t₀, t₁).
const PAPER = (() => {
  const alphas = Array.from({ length: 10 }, (_, i) => +(0.90 + 0.01 * i).toFixed(2));
  const sets = {
    mess3:  [{ x: 0.01, a: 0.005 }, { x: 0.02, a: 0.005 },
             ...[0.01, 0.05, 0.10, 0.60, 0.70, 0.80, 0.85, 0.90].map(a => ({ x: 0.02, a }))],
    arch:   alphas.map(a => ({ a })),
    wing:   alphas.map(x => ({ x, y: 0.4 })),
    strata: alphas.map(a => ({ a, t0: 0.38, t1: 0.54 })),
  };
  // The one parametrization per family shown in the paper's main text.
  const example = {
    mess3: { x: 0.02, a: 0.01 }, arch: { a: 0.99 },
    wing: { x: 0.98, y: 0.4 }, strata: { a: 0.97, t0: 0.38, t1: 0.54 },
  };
  const same = (p, q) => Object.keys(p).every(k => Math.abs(p[k] - q[k]) < 1e-9);
  const points = [];
  for (const [proc, list] of Object.entries(sets)) {
    for (const params of list) {
      const m = processMetrics(proc, params);
      points.push({ proc, params, mu: m.mu, h: m.h_TM, example: same(params, example[proc]) });
    }
  }
  return { points, families: Object.keys(sets) };
})();

// The highlighted regime is the span of the 40 parametrizations themselves, on
// whichever entropy axis the current process uses. HMMs use H̃; GHMMs have no
// H̃, so there the paper's HMMs are placed by entropy rate (the GHMM axis).
const paperRegimeCache = {};
function paperRegime(ghmm) {
  const axis = ghmm ? 'rate' : 'h';
  if (paperRegimeCache[axis]) return paperRegimeCache[axis];
  const hOf = ghmm
    ? (pt) => (pt.rate ??= entropyRate(buildModel(pt.proc, pt.params)) / Math.log(PROCESSES[pt.proc].symbols))
    : (pt) => pt.h;
  const span = (f) => {
    const v = PAPER.points.map(f);
    return [Math.min(...v), Math.max(...v)];
  };
  const box = { mu: span(pt => pt.mu), h: span(hOf) };
  const inside = (mu, h) => mu >= box.mu[0] && mu <= box.mu[1] && h >= box.h[0] && h <= box.h[1];
  return (paperRegimeCache[axis] = { hOf, box, inside });
}

// ── Sampling sizes (log-spaced steps; points are capped) ───────────────────────
const BATCH_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
const SEQ_STEPS   = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
const MAX_POINTS  = 500000;   // the paper's Fig. 9 uses 50 × 10,000

// ── Plot theme ─────────────────────────────────────────────────────────────────
const BG       = '#08090c';
const PLOT_BG  = '#0f1014';
const GRID_COL = '#1a1b20';
const TEXT_COL = '#66635b';
const TICK_COL = '#8a8478';
const INK      = '#e8e0d4';
const ACCENT   = '#ddb070';
const SYMBOL_COLORS = ['#c9944a', '#5b9a8b', '#8bba7f', '#d4685a', '#ddb070'];

// ── DOM + state ────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
let DOM;

let currentProcess = 'mess3';
let currentMode    = 'sample';      // 'sample' | 'enumerate'
let currentView    = 'geometry';    // 'geometry' | 'sweep' | 'emissions'
let lastData       = null;
let lastParams     = null;
let isComputing    = false;
let debounceTimer  = null;
let loadingTimer   = null;
let seedCounter    = 0;

const sweepState = { color: 'mu', scale: {}, paper: true, axes: {} };
const sweepScaleFor = (proc) => sweepState.scale[proc] ?? PROCESSES[proc].sweepScale ?? 'linear';
let sweepCache = { key: null, data: null };

// ── Boot ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  DOM = {
    paramList:   $('param-list'),
    paramSec:    $('param-section'),
    desc:        $('description'),
    meta:        $('process-meta'),
    caption:     $('view-caption'),
    computeBtn:  $('compute-btn'),
    loadingOv:   $('loading-overlay'),
    mainPlot:    $('main-plot'),
    batchSl:     $('batch-size'),
    batchVal:    $('batch-size-val'),
    seqSl:       $('seq-len'),
    seqVal:      $('seq-len-val'),
    pointEst:    $('point-estimate'),
    sampleSec:   $('sample-section'),
    depthSl:     $('max-seq-len'),
    depthVal:    $('max-seq-len-val'),
    nodeEst:     $('node-estimate'),
    enumSec:     $('enum-section'),
    samplingSum: $('sampling-summary'),
    modeSample:  $('mode-sample'),
    modeEnum:    $('mode-enum'),
    sweepAxes:   $('sweep-axes'),
    sweepColor:  $('sweep-color'),
    sweepScale:  $('sweep-scale'),
    paperToggle: $('paper-toggle'),
    readout:     $('sweep-readout'),
    metricsList: $('metrics-list'),
    paramPlot:   $('param-plot'),
    miniGeo:     $('mini-geo'),
    miniPlot:    $('mini-plot'),
    regimePlot:  $('regime-plot'),
    blochView:     $('bloch-view'),
    blochDpad:     $('bloch-dpad'),
    blochPlay:     $('bloch-play'),
    blochXZ:       $('bloch-xz'),
    blochR:        $('bloch-r'),
    blochTokens:   $('bloch-tokens'),
    blochSC:       $('bloch-sc'),
    blochFlipNote: $('bloch-flip-note'),
  };

  document.querySelectorAll('.proc-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      if (btn.dataset.proc !== currentProcess) selectProcess(btn.dataset.proc);
    }));
  document.querySelectorAll('.view-tab').forEach(btn =>
    btn.addEventListener('click', () => setView(btn.dataset.view)));

  DOM.modeSample.addEventListener('click', () => setMode('sample'));
  DOM.modeEnum.addEventListener('click', () => setMode('enumerate'));
  DOM.computeBtn.addEventListener('click', compute);

  DOM.batchSl.addEventListener('input', () => { capPoints('batch'); updateSampling(); scheduleAutoCompute(); });
  DOM.seqSl.addEventListener('input',   () => { capPoints('seq');   updateSampling(); scheduleAutoCompute(); });
  DOM.depthSl.addEventListener('input', updateSampling);

  setupSweepControls();
  setupBlochControls();
  DOM.miniPlot.addEventListener('click', () => setView('geometry'));
  DOM.miniPlot.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setView('geometry'); }
  });
  setupPlotAutoFit();

  const { proc, view } = parseHash();
  if (view) setView(view, false);
  selectProcess(proc || 'mess3');
  window.addEventListener('hashchange', () => {
    const h = parseHash();
    if (h.view && h.view !== currentView) setView(h.view, false);
    if (h.proc && h.proc !== currentProcess) selectProcess(h.proc);
  });
});

// ── URL hash: #process or #process/view ───────────────────────────────────────
function parseHash() {
  const [proc, view] = location.hash.slice(1).split('/');
  return {
    proc: PROCESSES[proc] ? proc : null,
    view: ['geometry', 'sweep', 'emissions'].includes(view) ? view : null,
  };
}

function writeHash() {
  const suffix = currentView === 'geometry' ? '' : `/${currentView}`;
  history.replaceState(null, '', `#${currentProcess}${suffix}`);
}

// ── Views ──────────────────────────────────────────────────────────────────────
function setView(view, render = true) {
  currentView = view;
  document.querySelectorAll('.view-tab').forEach(b => {
    const on = b.dataset.view === view;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on);
  });
  ['geometry', 'sweep', 'emissions'].forEach(v => $(`view-${v}`).classList.toggle('hidden', v !== view));
  DOM.miniGeo.classList.toggle('hidden', view === 'geometry');
  if (view !== 'geometry') blochSetPlaying(false);
  writeHash();
  if (render && lastData) renderActiveView();
}

function renderActiveView() {
  if (currentView === 'geometry') return renderGeometry(lastData, lastParams);
  if (currentView === 'sweep') renderSweep(lastParams, lastData.metrics);
  else renderEmissions(lastData);
  renderMiniGeometry(lastData);
}

// ── Mode / sampling ───────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  DOM.modeSample.classList.toggle('active', mode === 'sample');
  DOM.modeEnum.classList.toggle('active', mode === 'enumerate');
  DOM.sampleSec.classList.toggle('hidden', mode !== 'sample');
  DOM.enumSec.classList.toggle('hidden', mode !== 'enumerate');
  DOM.computeBtn.querySelector('.btn-label').textContent = mode === 'sample' ? 'Resample' : 'Enumerate';
  updateSampling();
}

const batchSize = () => BATCH_STEPS[+DOM.batchSl.value];
const seqLen    = () => SEQ_STEPS[+DOM.seqSl.value];

// Keep batch × (length + 1) ≤ MAX_POINTS by lowering the slider not being moved.
function capPoints(moved) {
  const other = moved === 'batch' ? DOM.seqSl : DOM.batchSl;
  while (+other.value > 0 && batchSize() * (seqLen() + 1) > MAX_POINTS) other.value = +other.value - 1;
}

function updateSampling() {
  DOM.batchVal.textContent = batchSize().toLocaleString();
  DOM.seqVal.textContent   = seqLen().toLocaleString();
  DOM.pointEst.textContent = (batchSize() * (seqLen() + 1)).toLocaleString();
  DOM.depthVal.textContent = DOM.depthSl.value;

  const depth = parseInt(DOM.depthSl.value);
  const cfg = PROCESSES[currentProcess];
  let n = 0;
  if (cfg.nodeEstimate) n = cfg.nodeEstimate(depth);
  else for (let d = 0; d <= depth; d++) n += cfg.symbols ** d;
  DOM.nodeEst.textContent = n.toLocaleString();

  DOM.samplingSum.textContent = currentMode === 'sample'
    ? `${batchSize().toLocaleString()} × ${seqLen().toLocaleString()}`
    : `depth ${depth}`;
}

// ── Process selection ──────────────────────────────────────────────────────────
function selectProcess(proc, values = null) {
  currentProcess = proc;
  document.querySelectorAll('.proc-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.proc === proc));

  const cfg = PROCESSES[proc];
  renderParams(cfg, values);
  blochSetPlaying(false);
  writeHash();

  // Enumeration grows as symbols^depth, so wide alphabets get a lower cap.
  DOM.depthSl.max = cfg.maxDepth ?? 14;

  DOM.desc.innerHTML = cfg.description;
  DOM.meta.textContent = cfg.meta
    ?? `${cfg.states} states · ${cfg.symbols} symbols · ${cfg.ghmm ? 'GHMM' : 'HMM'}`;
  updateSampling();
  compute();
}

// ── Parameter panel ────────────────────────────────────────────────────────────
const decsFor = (p) => p.step < 0.01 ? 3 : (p.step < 0.1 ? 2 : (p.step < 10 ? 1 : 0));

function renderParams(cfg, values) {
  DOM.paramSec.classList.toggle('hidden', !cfg.params.length);
  DOM.paramList.innerHTML = cfg.params.map(p => {
    const v = values?.[p.id] ?? p.default;
    return `
      <div class="param-row">
        <div class="param-header">
          <label class="param-label" for="slider-${p.id}">${p.label}</label>
          <input type="number" class="param-val param-num" id="val-${p.id}" aria-label="${p.label}"
                 min="${p.min}" max="${p.max}" step="${p.step}" value="${v.toFixed(decsFor(p))}" />
        </div>
        <input type="range" class="slider" id="slider-${p.id}"
               min="${p.min}" max="${p.max}" step="${p.step}" value="${v}" />
      </div>`;
  }).join('');

  cfg.params.forEach(p => {
    const sl = $(`slider-${p.id}`), vEl = $(`val-${p.id}`);
    const decs = decsFor(p);
    const clamp = (v) => Math.min(p.max, Math.max(p.min, v));
    sl.addEventListener('input', () => {
      vEl.value = parseFloat(sl.value).toFixed(decs);
      scheduleAutoCompute();
    });
    vEl.addEventListener('input', () => {
      const v = parseFloat(vEl.value);
      if (Number.isFinite(v)) { sl.value = clamp(v); scheduleAutoCompute(); }
    });
    vEl.addEventListener('change', () => {
      const v = parseFloat(vEl.value);
      const c = Number.isFinite(v) ? clamp(v) : parseFloat(sl.value);
      sl.value = c;
      vEl.value = c.toFixed(decs);
    });
  });
}

function getParams() {
  return Object.fromEntries(PROCESSES[currentProcess].params.map(p => {
    const sl = $(`slider-${p.id}`);
    return [p.id, sl ? parseFloat(sl.value) : p.default];
  }));
}

// Set several parameters at once (sweep clicks, paper points), then recompute.
function setParams(values) {
  for (const p of PROCESSES[currentProcess].params) {
    if (!(p.id in values)) continue;
    const v = Math.min(p.max, Math.max(p.min, values[p.id]));
    $(`slider-${p.id}`).value = v;
    $(`val-${p.id}`).value = parseFloat($(`slider-${p.id}`).value).toFixed(decsFor(p));
  }
  compute();
}

function scheduleAutoCompute() {
  clearTimeout(debounceTimer);
  if (currentMode === 'sample') debounceTimer = setTimeout(compute, 150);
}

// ── Compute ────────────────────────────────────────────────────────────────────
function compute() {
  if (isComputing) return;
  clearTimeout(debounceTimer);
  isComputing = true;
  DOM.computeBtn.disabled = true;
  clearTimeout(loadingTimer);
  loadingTimer = setTimeout(() => DOM.loadingOv.classList.remove('hidden'), 120);

  const proc = currentProcess;
  const params = getParams();

  // setTimeout(0) lets the loading overlay paint before the sync computation.
  setTimeout(() => {
    try {
      const modeParams = currentMode === 'sample'
        ? { batchSize: batchSize(), seqLen: seqLen(), seed: seedCounter++ }
        : { maxSeqLen: parseInt(DOM.depthSl.value) };
      const data = computeResult(proc, params, currentMode, modeParams);
      if (proc !== currentProcess) return;
      lastData = data;
      lastParams = params;
      DOM.caption.textContent = buildCaption(proc, params, data);
      renderActiveView();
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

function buildCaption(proc, params, data) {
  const cfg = PROCESSES[proc];
  const parts = cfg.params.map(p => {
    const name = p.label.split(' ')[0];
    return `${name} = ${(params[p.id] ?? p.default).toFixed(Math.min(decsFor(p), 2))}`;
  });
  if (proc === 'bloch') parts.push(`s = ${data.s.toFixed(3)}`);
  parts.push(`${data.n.toLocaleString()} ${data.mode === 'sample' ? 'sampled' : 'enumerated'}`);
  return parts.join('  ·  ');
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

// Panels are shown and hidden while the plots render, so a plot drawn first
// can be sized for a layout that no longer holds. Re-fit whenever a plot's box
// changes (Plotly's own `responsive` only listens for window resizes).
function setupPlotAutoFit() {
  const observer = new ResizeObserver(entries => {
    for (const { target: el } of entries) {
      const fl = el._fullLayout;
      if (!fl || el.offsetParent === null) continue;
      if (fl.width !== el.clientWidth || fl.height !== el.clientHeight) Plotly.Plots.resize(el);
    }
  });
  document.querySelectorAll('.viz').forEach(el => observer.observe(el));
}

function plotConfig() {
  return {
    responsive: true, displayModeBar: 'hover', displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'autoScale2d'],
  };
}

const axisStyle = (title) => ({
  title: title ? { text: title, font: { size: 11, color: TEXT_COL }, standoff: 6 } : undefined,
  color: TEXT_COL, gridcolor: GRID_COL, zeroline: false, tickfont: { size: 10, color: TICK_COL },
});

function traceType(n) { return n > 2000 ? 'scattergl' : 'scatter'; }

// ══════════════════════════════════════════════════════════════════════
//  GEOMETRY VIEW
// ══════════════════════════════════════════════════════════════════════

function renderGeometry(data, params) {
  const isBloch = data.type === 'bloch';
  DOM.blochView.classList.toggle('hidden', !isBloch);
  DOM.mainPlot.classList.toggle('hidden', isBloch);
  if (isBloch) renderBloch(data, params);
  else if (data.type === 'simplex') plotSimplex(data, 'main-plot');
  else plotPCA(data, 'main-plot');
}

function plotSimplex(data, containerId) {
  const sqrt3 = Math.sqrt(3);
  const V = [[0, 0], [1, 0], [0.5, sqrt3 / 2]];
  const n = data.n;
  const colors = new Array(n);
  for (let i = 0; i < n; i++) colors[i] = `rgb(${data.r[i]},${data.g[i]},${data.b[i]})`;

  const edges = {
    x: [V[0][0], V[1][0], V[2][0], V[0][0]], y: [V[0][1], V[1][1], V[2][1], V[0][1]],
    mode: 'lines', type: 'scatter', line: { color: '#2e3040', width: 1.5 },
    hoverinfo: 'skip',
  };
  const vertices = {
    x: V.map(v => v[0]), y: V.map(v => v[1]),
    mode: 'markers+text', type: 'scatter',
    text: ['S₀', 'S₁', 'S₂'],
    textposition: ['bottom left', 'bottom right', 'top center'],
    textfont: { size: 13, color: TEXT_COL, family: 'Fira Code, monospace' },
    marker: { color: TEXT_COL, size: 7 },
    hoverinfo: 'skip',
  };
  const pts = {
    x: data.x, y: data.y,
    mode: 'markers', type: traceType(n),
    marker: { color: colors, size: Math.max(1.5, Math.min(5, Math.round(8000 / n))), opacity: 0.88 },
    customdata: data.bs,
    hovertemplate: 'P(S₀) %{customdata[0]:.4f}<br>P(S₁) %{customdata[1]:.4f}<br>P(S₂) %{customdata[2]:.4f}<extra></extra>',
  };
  const layout = {
    xaxis: { visible: false, range: [-0.1, 1.1], scaleanchor: 'y' },
    yaxis: { visible: false, range: [-0.12, sqrt3 / 2 + 0.1] },
    paper_bgcolor: BG, plot_bgcolor: BG,
    margin: { t: 16, b: 16, l: 16, r: 16 },
    showlegend: false, autosize: true,
  };
  Plotly.react(containerId, [edges, vertices, pts], layout, plotConfig());
}

function plotPCA(data, containerId) {
  const [vr1, vr2] = data.var_ratios;
  const n = data.n;
  const pts = {
    x: data.x, y: data.y,
    mode: 'markers', type: traceType(n),
    marker: {
      color: data.color_val, colorscale: 'RdBu', reversescale: true,
      size: Math.max(1.5, Math.min(5, Math.round(8000 / n))), opacity: 0.88,
      colorbar: {
        title: { text: 'P(next = 0)', font: { color: TICK_COL, size: 11 } },
        thickness: 10, len: 0.6, tickfont: { color: TICK_COL, size: 10 },
        bgcolor: 'rgba(0,0,0,0)', outlinewidth: 0, tickformat: '.2f',
      },
    },
    hovertemplate: 'P(next = 0) %{marker.color:.4f}<extra></extra>',
  };
  const layout = {
    xaxis: axisStyle(`PC1 (${(vr1 * 100).toFixed(0)}%)`),
    yaxis: axisStyle(`PC2 (${(vr2 * 100).toFixed(0)}%)`),
    paper_bgcolor: BG, plot_bgcolor: BG,
    margin: { t: 16, b: 48, l: 56, r: 16 },
    showlegend: false, autosize: true,
  };
  Plotly.react(containerId, [pts], layout, plotConfig());
}

// Small, static copy of the geometry for the Sweep and Emissions views.
const MINI_MAX_POINTS = 20000;

function renderMiniGeometry(data) {
  const stride = Math.max(1, Math.ceil(data.n / MINI_MAX_POINTS));
  const pick = (arr) => arr.filter((_, i) => i % stride === 0);
  const marker = { size: stride > 1 || data.n > 5000 ? 1.5 : 2.5, opacity: 0.85 };
  const traces = [];
  const shapes = [];
  let xr, yr;

  if (data.type === 'simplex') {
    const h = Math.sqrt(3) / 2;
    traces.push({ x: [0, 1, 0.5, 0], y: [0, 0, h, 0], mode: 'lines', type: 'scatter',
                  line: { color: '#2e3040', width: 1 } });
    const r = pick(data.r), g = pick(data.g), b = pick(data.b);
    traces.push({ x: pick(data.x), y: pick(data.y), mode: 'markers', type: 'scattergl',
                  marker: { ...marker, color: r.map((v, i) => `rgb(${v},${g[i]},${b[i]})`) } });
    xr = [-0.04, 1.04]; yr = [-0.04, h + 0.04];
  } else if (data.type === 'bloch') {
    shapes.push({ type: 'circle', x0: -1, y0: -1, x1: 1, y1: 1, line: { color: '#2e3040', width: 1 } });
    traces.push({ x: pick(data.x), y: pick(data.z), mode: 'markers', type: 'scattergl',
                  marker: { ...marker, color: pick(data.t),
                            colorscale: [[0, '#2e3040'], [0.4, '#5b9a8b'], [1, '#ddb070']] } });
    xr = [-1.06, 1.06]; yr = [-1.06, 1.06];
  } else {
    traces.push({ x: pick(data.x), y: pick(data.y), mode: 'markers', type: 'scattergl',
                  marker: { ...marker, color: pick(data.color_val), colorscale: 'RdBu', reversescale: true } });
  }

  Plotly.react(DOM.miniPlot, traces, {
    xaxis: { visible: false, range: xr, scaleanchor: 'y' },
    yaxis: { visible: false, range: yr },
    shapes,
    paper_bgcolor: BG, plot_bgcolor: BG,
    margin: { t: 8, b: 8, l: 8, r: 8 },
    showlegend: false, autosize: true,
  }, { staticPlot: true, responsive: true });
}

// ══════════════════════════════════════════════════════════════════════
//  EMISSIONS VIEW (distribution of next-token probabilities)
// ══════════════════════════════════════════════════════════════════════

const RUG_MAX_POINTS = 1500;

function subsampleSorted(arr) {
  const sorted = Float64Array.from(arr).sort();
  if (sorted.length <= RUG_MAX_POINTS) return Array.from(sorted);
  const out = new Float64Array(RUG_MAX_POINTS);
  for (let i = 0; i < RUG_MAX_POINTS; i++) {
    out[i] = sorted[Math.round(i * (sorted.length - 1) / (RUG_MAX_POINTS - 1))];
  }
  return Array.from(out);
}

function renderEmissions(data) {
  const { obsProbs, numSymbols } = data;
  plotRug('obs-prob-plot', 'P(next token)', obsProbs, numSymbols, [-0.02, 1.02]);
  let minLog = 0;
  const logProbs = obsProbs.map(arr => arr.map(v => {
    const l = v > 0 ? Math.log(v) : -20;
    if (l < minLog) minLog = l;
    return l;
  }));
  plotRug('log-obs-prob-plot', 'log P(next token)', logProbs, numSymbols,
          [Math.min(minLog * 1.1, -0.5), 0.1]);
}

function plotRug(containerId, title, values, numSymbols, range) {
  const traces = [];
  for (let s = 0; s < numSymbols; s++) {
    const xs = subsampleSorted(values[s]);
    traces.push({
      x: xs, y: new Array(xs.length).fill(s),
      mode: 'markers', type: 'scatter', name: `token ${s}`,
      marker: { color: SYMBOL_COLORS[s], size: 18, symbol: 'line-ns', opacity: 0.7,
                line: { width: 1.2, color: SYMBOL_COLORS[s] } },
      hovertemplate: `%{x:.4f}<extra>token ${s}</extra>`,
    });
  }
  const layout = {
    title: { text: title, font: { size: 11, color: TICK_COL }, x: 0.01, xanchor: 'left', y: 0.97 },
    xaxis: { ...axisStyle(), range },
    yaxis: {
      ...axisStyle(), range: [-0.7, numSymbols - 0.3],
      tickvals: [...Array(numSymbols).keys()], ticktext: [...Array(numSymbols).keys()].map(String),
      gridcolor: 'rgba(0,0,0,0)',
    },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 32, b: 32, l: 32, r: 16 },
    showlegend: false, autosize: true,
  };
  Plotly.react(containerId, traces, layout, plotConfig());
}

// ══════════════════════════════════════════════════════════════════════
//  SWEEP VIEW (mixing rate and entropy over parameter space)
// ══════════════════════════════════════════════════════════════════════

function sweepAxesFor(proc) {
  const cfg = PROCESSES[proc];
  const ids = cfg.params.map(p => p.id);
  if (cfg.sweepAxes && ids.length < 3) return cfg.sweepAxes;
  if (ids.length <= 2) return ids;
  return sweepState.axes[proc] ?? cfg.sweepAxes ?? ids.slice(0, 2);
}

function gridValues(p, n, log) {
  const lo = log ? Math.max(p.min, 1e-3) : p.min;
  return Array.from({ length: n }, (_, i) => {
    const t = i / (n - 1);
    return log ? lo * Math.pow(p.max / lo, t) : lo + t * (p.max - lo);
  });
}

function computeSweep(proc, axes, params) {
  const cfg = PROCESSES[proc];
  const log = sweepScaleFor(proc) === 'log';
  const fixed = Object.fromEntries(Object.entries(params).filter(([k]) => !axes.includes(k)));
  const key = JSON.stringify([proc, axes, log, fixed]);
  if (sweepCache.key === key) return sweepCache.data;

  const entropyOf = (m) => (cfg.ghmm ? m.h_rate_tilde : m.h_TM);
  const pdef = (id) => cfg.params.find(p => p.id === id);
  let data;
  if (axes.length === 0) {
    data = { dim: 0 };
  } else if (axes.length === 1) {
    const xs = gridValues(pdef(axes[0]), cfg.ghmm ? 61 : 121, log);
    const ms = xs.map(v => processMetrics(proc, { ...params, [axes[0]]: v }));
    data = { dim: 1, xs, mu: ms.map(m => m.mu), h: ms.map(entropyOf) };
  } else {
    const N = 61;
    const xs = gridValues(pdef(axes[0]), N, log), ys = gridValues(pdef(axes[1]), N, log);
    const mu = [], h = [];
    for (const yv of ys) {
      const rowMu = [], rowH = [];
      for (const xv of xs) {
        const m = processMetrics(proc, { ...params, [axes[0]]: xv, [axes[1]]: yv });
        rowMu.push(m ? m.mu : NaN);
        rowH.push(m ? entropyOf(m) : NaN);
      }
      mu.push(rowMu); h.push(rowH);
    }
    data = { dim: 2, xs, ys, mu, h };
  }
  sweepCache = { key, data };
  return data;
}

// Paper parametrizations of this family that lie on the current slice.
function paperOnSlice(proc, axes, params) {
  return PAPER.points
    .map((pt, i) => ({ ...pt, i }))
    .filter(pt => pt.proc === proc &&
      Object.keys(pt.params).every(k => axes.includes(k) || Math.abs(pt.params[k] - params[k]) < 1e-6));
}

function paperLabel(pt) {
  const cfg = PROCESSES[pt.proc];
  const ps = Object.entries(pt.params)
    .map(([k, v]) => `${cfg.params.find(p => p.id === k).label.split(' ')[0]}=${v}`).join(', ');
  return `${cfg.label}  ${ps}${pt.example ? '  ★ main text' : ''}`;
}

function loadPaperPoint(i) {
  const pt = PAPER.points[i];
  if (pt.proc !== currentProcess) selectProcessWith(pt.proc, pt.params);
  else setParams(pt.params);
}

function selectProcessWith(proc, values) {
  const cfg = PROCESSES[proc];
  selectProcess(proc, Object.fromEntries(cfg.params.map(p => [p.id, values[p.id] ?? p.default])));
}

function setupSweepControls() {
  DOM.sweepColor.querySelectorAll('.seg-btn').forEach(btn => btn.addEventListener('click', () => {
    sweepState.color = btn.dataset.color;
    DOM.sweepColor.querySelectorAll('.seg-btn').forEach(b => b.classList.toggle('active', b === btn));
    renderSweep(lastParams, lastData.metrics);
  }));
  DOM.sweepScale.querySelectorAll('.seg-btn').forEach(btn => btn.addEventListener('click', () => {
    sweepState.scale[currentProcess] = btn.dataset.scale;
    renderSweep(lastParams, lastData.metrics);
  }));
  DOM.paperToggle.addEventListener('click', () => {
    sweepState.paper = !sweepState.paper;
    DOM.paperToggle.setAttribute('aria-pressed', sweepState.paper);
    renderSweep(lastParams, lastData.metrics);
  });

  const onClick = (ev) => {
    const pt = ev.points?.[0];
    if (!pt) return;
    if (pt.data.name === 'paper') return loadPaperPoint(pt.customdata);
    const axes = sweepAxesFor(currentProcess);
    if (pt.data.name === 'grid' && axes.length === 2) setParams({ [axes[0]]: pt.x, [axes[1]]: pt.y });
    else if (pt.data.name === 'curve' && axes.length === 1) setParams({ [axes[0]]: pt.x });
  };
  for (const el of [DOM.paramPlot, DOM.regimePlot]) {
    Plotly.newPlot(el, [], { paper_bgcolor: BG, plot_bgcolor: BG }, plotConfig());
    el.on('plotly_click', onClick);
  }
}

function renderSweepAxisPickers(proc, axes) {
  const cfg = PROCESSES[proc];
  if (cfg.params.length < 3) { DOM.sweepAxes.innerHTML = ''; return; }
  const opts = (sel) => cfg.params.map(p =>
    `<option value="${p.id}"${p.id === sel ? ' selected' : ''}>${p.label}</option>`).join('');
  DOM.sweepAxes.innerHTML =
    `<label>x <select id="axis-x">${opts(axes[0])}</select></label>` +
    `<label>y <select id="axis-y">${opts(axes[1])}</select></label>`;
  const update = () => {
    let x = $('axis-x').value, y = $('axis-y').value;
    if (x === y) y = cfg.params.find(p => p.id !== x).id;
    sweepState.axes[proc] = [x, y];
    renderSweep(lastParams, lastData.metrics);
  };
  $('axis-x').addEventListener('change', update);
  $('axis-y').addEventListener('change', update);
}

function renderSweep(params, metrics) {
  const proc = currentProcess;
  const cfg = PROCESSES[proc];
  const axes = sweepAxesFor(proc);
  const sweep = computeSweep(proc, axes, params);
  const showPaper = sweepState.paper;
  const regime = paperRegime(!!cfg.ghmm);
  const hCur = cfg.ghmm ? metrics.h_rate_tilde : metrics.h_TM;

  DOM.sweepColor.classList.toggle('hidden', sweep.dim !== 2);
  DOM.sweepScale.classList.toggle('hidden', sweep.dim === 0);
  DOM.sweepScale.querySelectorAll('.seg-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.scale === sweepScaleFor(proc)));
  DOM.paramPlot.classList.toggle('hidden', sweep.dim === 0);
  renderSweepAxisPickers(proc, axes);
  renderReadout(metrics, hCur, showPaper && regime);

  if (sweep.dim === 2) plotSweep2D(sweep, axes, params, showPaper && regime);
  else if (sweep.dim === 1) plotSweep1D(sweep, axes[0], params, metrics, hCur, showPaper && regime);
  plotRegime(sweep, metrics.mu, hCur, showPaper && regime, cfg.ghmm);
}

function renderReadout(m, hCur, regime) {
  const hName = m.ghmm ? 'entropy rate' : 'H̃';
  let html = `<span>|λ₂| <b>${fmtFixed(m.mu)}</b></span><span>${hName} <b>${fmtFixed(hCur)}</b></span>`;
  if (regime) {
    const inside = regime.inside(m.mu, hCur);
    html += `<span class="regime-chip${inside ? ' in' : ''}">${inside ? 'in paper regime' : 'outside paper regime'}</span>`;
  }
  DOM.readout.innerHTML = html;

  const rows = m.ghmm ? [
    ['λ₂', fmtComplex(m.lambda2_raw)],
    ['entropy rate', `${fmtFixed(m.h_rate)} nats`],
    ['entropy rate / log n', fmtFixed(m.h_rate_tilde)],
    ['1 / (1 − |λ₂|)', `${fmtFixed(m.mixingTime, 2)} steps`],
  ] : [
    ['λ₂', fmtComplex(m.lambda2_raw)],
    ['π', '(' + m.pi.map(p => p.toFixed(3)).join(', ') + ')'],
    ['H_T, H_M', `${fmtFixed(m.H_A)}, ${fmtFixed(m.H_B)} nats`],
    ['H̃_T, H̃_M', `${fmtFixed(m.h_A_tilde)}, ${fmtFixed(m.h_B_tilde)}`],
    ['H̃ = (H_T + H_M)/(log m + log n)', fmtFixed(m.h_TM)],
    ['h_total = H̃_T + H̃_M', fmtFixed(m.h_total)],
    ['1 / (1 − |λ₂|)', `${fmtFixed(m.mixingTime, 2)} steps`],
  ];
  DOM.metricsList.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}

const MU_SCALE = 'YlOrRd';
const H_SCALE  = 'Viridis';
const labelOf  = (id) => PROCESSES[currentProcess].params.find(p => p.id === id).label;

function paperMarkers(points, xOf, yOf) {
  return {
    name: 'paper',
    x: points.map(xOf), y: points.map(yOf),
    mode: 'markers', type: 'scatter',
    marker: {
      symbol: points.map(p => (p.example ? 'star' : 'diamond')),
      size: points.map(p => (p.example ? 15 : 9)),
      color: ACCENT, line: { color: BG, width: 1.2 },
    },
    customdata: points.map(p => p.i),
    text: points.map(paperLabel),
    hovertemplate: '%{text}<br>click to load<extra>paper</extra>',
  };
}

// The family is in the paper but none of its parametrizations lie on this slice.
function offSliceNote(proc, onSlice) {
  if (onSlice.length || !PAPER.points.some(p => p.proc === proc)) return [];
  return [{
    x: 0.5, y: 1, xref: 'paper', yref: 'paper', yanchor: 'bottom', showarrow: false,
    text: 'Paper points are off this slice.<br>Click one on the right to load it.',
    font: { size: 11, color: TICK_COL },
  }];
}

function plotSweep2D(sweep, axes, params, regime) {
  const isMu = sweepState.color === 'mu';
  const z = isMu ? sweep.mu : sweep.h;
  const heat = {
    name: 'grid', type: 'heatmap', z, x: sweep.xs, y: sweep.ys,
    zmin: 0, zmax: 1,
    colorscale: isMu ? MU_SCALE : H_SCALE,
    colorbar: {
      title: { text: isMu ? '|λ₂|' : 'H̃', font: { color: TICK_COL, size: 11 } },
      thickness: 8, len: 0.7, outlinewidth: 0, tickfont: { color: TICK_COL, size: 9 },
    },
    hovertemplate: `${labelOf(axes[0])} %{x:.3f}<br>${labelOf(axes[1])} %{y:.3f}<br>` +
                   `${isMu ? '|λ₂|' : 'H̃'} %{z:.3f}<br>click to set<extra></extra>`,
  };
  const traces = [heat];
  let annotations = [];

  if (regime) {
    // Outline the cells whose (|λ₂|, entropy) fall inside the paper's region.
    const inside = sweep.mu.map((row, i) => row.map((mu, j) => (regime.inside(mu, sweep.h[i][j]) ? 1 : 0)));
    traces.push({
      type: 'contour', z: inside, x: sweep.xs, y: sweep.ys,
      contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
      line: { color: INK, width: 1.5, dash: 'dot' },
      showscale: false, hoverinfo: 'skip',
    });
    const pts = paperOnSlice(currentProcess, axes, params);
    if (pts.length) traces.push(paperMarkers(pts, p => p.params[axes[0]], p => p.params[axes[1]]));
    annotations = offSliceNote(currentProcess, pts);
  }
  traces.push({
    x: [params[axes[0]]], y: [params[axes[1]]], mode: 'markers', type: 'scatter',
    marker: { size: 12, color: 'rgba(0,0,0,0)', line: { color: INK, width: 2 } },
    hoverinfo: 'skip',
  });

  const log = sweepScaleFor(currentProcess) === 'log';
  Plotly.react(DOM.paramPlot, traces, {
    xaxis: { ...axisStyle(labelOf(axes[0])), type: log ? 'log' : 'linear' },
    yaxis: { ...axisStyle(labelOf(axes[1])), type: log ? 'log' : 'linear' },
    annotations,
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: annotations.length ? 40 : 16, b: 44, l: 48, r: 8 },
    showlegend: false, autosize: true,
  }, plotConfig());
}

function plotSweep1D(sweep, axis, params, metrics, hCur, regime) {
  const cfg = PROCESSES[currentProcess];
  const x0 = params[axis];
  const hName = cfg.ghmm ? 'entropy rate' : 'H̃';
  const traces = [
    { name: 'curve', x: sweep.xs, y: sweep.mu, mode: 'lines', type: 'scatter',
      line: { color: SYMBOL_COLORS[0], width: 2 },
      hovertemplate: '|λ₂| %{y:.3f}<extra></extra>' },
    { name: 'curve', x: sweep.xs, y: sweep.h, mode: 'lines', type: 'scatter',
      line: { color: SYMBOL_COLORS[1], width: 2 },
      hovertemplate: `${hName} %{y:.3f}<extra></extra>` },
    { x: [x0, x0], y: [metrics.mu, hCur], mode: 'markers', type: 'scatter',
      marker: { size: 9, color: [SYMBOL_COLORS[0], SYMBOL_COLORS[1]], line: { color: BG, width: 1.5 } },
      hoverinfo: 'skip' },
  ];
  const shapes = [{
    type: 'line', x0, x1: x0, y0: 0, y1: 1, yref: 'paper',
    line: { color: INK, width: 1, dash: 'dot' },
  }];

  if (regime) {
    // Shade the parameter ranges whose (|λ₂|, entropy) fall inside the paper's region.
    let start = null;
    sweep.xs.forEach((x, i) => {
      const inside = regime.inside(sweep.mu[i], sweep.h[i]);
      if (inside && start === null) start = x;
      if ((!inside || i === sweep.xs.length - 1) && start !== null) {
        shapes.push({ type: 'rect', layer: 'below', x0: start, x1: inside ? x : sweep.xs[i - 1],
                      y0: 0, y1: 1, yref: 'paper', fillcolor: 'rgba(232,224,212,0.07)', line: { width: 0 } });
        start = null;
      }
    });
    const pts = paperOnSlice(currentProcess, [axis], params);
    if (pts.length) traces.push(paperMarkers(pts, p => p.params[axis], p => p.mu));
  }

  Plotly.react(DOM.paramPlot, traces, {
    xaxis: { ...axisStyle(labelOf(axis)), type: sweepScaleFor(currentProcess) === 'log' ? 'log' : 'linear' },
    yaxis: { ...axisStyle(), range: [0, 1.04] },
    shapes,
    annotations: [
      { x: 0, y: 1.0, xref: 'paper', yref: 'paper', xanchor: 'left', yanchor: 'bottom', showarrow: false,
        text: `<span style="color:${SYMBOL_COLORS[0]}">|λ₂|</span>   <span style="color:${SYMBOL_COLORS[1]}">${hName}</span>`,
        font: { size: 11 } },
    ],
    hovermode: 'x',
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 24, b: 44, l: 40, r: 12 },
    showlegend: false, autosize: true,
  }, plotConfig());
}

// Where this process (and the paper's 40 HMMs) sit in the entropy–mixing plane.
function plotRegime(sweep, muCur, hCur, regime, ghmm) {
  const traces = [];
  const shapes = [], annotations = [];

  if (sweep.dim === 2) {
    traces.push({
      x: sweep.h.flat(), y: sweep.mu.flat(), mode: 'markers', type: 'scattergl',
      marker: { size: 3, color: 'rgba(138,132,120,0.22)' }, hoverinfo: 'skip',
    });
  } else if (sweep.dim === 1) {
    traces.push({
      x: sweep.h, y: sweep.mu, mode: 'lines', type: 'scatter',
      line: { color: 'rgba(138,132,120,0.6)', width: 1.5 }, hoverinfo: 'skip',
    });
  }

  if (regime) {
    const [h0, h1] = regime.box.h, [m0, m1] = regime.box.mu;
    shapes.push({ type: 'rect', layer: 'below', x0: h0, x1: h1, y0: m0, y1: m1,
                  fillcolor: 'rgba(221,176,112,0.08)', line: { color: 'rgba(221,176,112,0.45)', width: 1 } });
    for (const g of [0.5, 0.75, 0.95, 0.99]) {
      shapes.push({ type: 'line', layer: 'below', x0: 0, x1: 1, xref: 'paper', y0: g, y1: g,
                    line: { color: '#2e3040', width: 1, dash: 'dash' } });
      annotations.push({ x: 1, xref: 'paper', y: g, xanchor: 'right', yanchor: 'bottom', showarrow: false,
                         text: `${g}`, font: { size: 9, color: TEXT_COL } });
    }
    const pts = PAPER.points.map((pt, i) => ({ ...pt, i }));
    const mine = pts.filter(p => p.proc === currentProcess);
    const others = pts.filter(p => p.proc !== currentProcess);
    traces.push({
      ...paperMarkers(others, regime.hOf, p => p.mu),
      marker: { symbol: 'diamond', size: 7, color: 'rgba(138,132,120,0.55)', line: { width: 0 } },
    });
    if (mine.length) traces.push(paperMarkers(mine, regime.hOf, p => p.mu));
  }

  traces.push({
    x: [hCur], y: [muCur], mode: 'markers', type: 'scatter',
    marker: { size: 12, color: 'rgba(0,0,0,0)', line: { color: INK, width: 2 } },
    hovertemplate: 'current<br>|λ₂| %{y:.3f}<br>%{x:.3f}<extra></extra>',
  });

  Plotly.react(DOM.regimePlot, traces, {
    xaxis: { ...axisStyle(ghmm ? 'entropy rate / log n' : 'H̃ = (H_T + H_M) / (log m + log n)'), range: [0, 1.03] },
    yaxis: { ...axisStyle('|λ₂|'), range: [0, 1.02] },
    shapes, annotations,
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 16, b: 44, l: 48, r: 12 },
    showlegend: false, autosize: true,
  }, plotConfig());
}

// ══════════════════════════════════════════════════════════════════════
//  BLOCH WALK (belief disk + step-through walker)
// ══════════════════════════════════════════════════════════════════════

const BLOCH_TOKEN_COLORS = SYMBOL_COLORS.slice(0, 4);   // 0:+z  1:−z  2:+x  3:−x
const BLOCH_TRAIL_MAX    = 80;
const BLOCH_AUTOPLAY_MAX = 200;
const BLOCH_PLAY_MS      = 280;

const blochWalk = {
  model: null,
  tokens: [],             // the walker's context
  path: [[1, 0, 0]],      // belief after each prefix of `tokens`
  cloudTrace: null,       // background beliefs from Sample / Enumerate
  meanPurity: null,       // ⟨|r|⟩ per context length (sample mode only)
  seqLen: 20,
  timer: null,
};

const blochRadius = (belief) => Math.hypot(belief[1], belief[2]);

function setupBlochControls() {
  DOM.blochDpad.querySelectorAll('[data-tok]').forEach(btn =>
    btn.addEventListener('click', () => blochStepToken(parseInt(btn.dataset.tok))));
  $('bloch-random').addEventListener('click', blochRandomStep);
  DOM.blochPlay.addEventListener('click', () => blochSetPlaying(!blochWalk.timer));
  $('bloch-undo').addEventListener('click', blochUndo);
  $('bloch-reset').addEventListener('click', blochReset);
  $('bloch-mirror').addEventListener('click', blochInvertBeta);

  const ARROW_TOKENS = { ArrowUp: 0, ArrowDown: 1, ArrowRight: 2, ArrowLeft: 3 };
  document.addEventListener('keydown', (e) => {
    if (currentProcess !== 'bloch' || currentView !== 'geometry' || !blochWalk.model) return;
    if (['INPUT', 'SELECT'].includes(document.activeElement?.tagName)) return;   // sliders own the arrows
    if (e.key in ARROW_TOKENS) {
      e.preventDefault();
      blochStepToken(ARROW_TOKENS[e.key]);
    } else if (e.key === 'Backspace') {
      e.preventDefault();
      blochUndo();
    }
  });

  const gd = $('bloch-plot');
  Plotly.newPlot(gd, [], { paper_bgcolor: BG, plot_bgcolor: BG }, plotConfig());
  gd.on('plotly_click', (ev) => {
    const pt = ev.points[0];
    if (pt?.data.name === 'next') blochStepToken(pt.customdata[0]);
  });
}

function renderBloch(data, params) {
  const beta = params.beta ?? PROCESSES.bloch.params[0].default;
  blochWalk.model = blochModel(1.0, beta);

  const n = data.n;
  blochWalk.cloudTrace = {
    x: data.x, y: data.z,
    mode: 'markers', type: 'scattergl',
    marker: {
      color: data.t,
      colorscale: [[0, '#2e3040'], [0.4, '#5b9a8b'], [1, '#ddb070']],
      size: Math.max(1.5, Math.min(5, Math.round(8000 / n))),
      opacity: 0.75,
      showscale: $('bloch-plot').clientWidth >= 520,   // no room beside a small disk
      colorbar: {
        title: { text: 'context length', font: { color: TICK_COL, size: 11 } },
        thickness: 8, len: 0.5, outlinewidth: 0,
        tickfont: { color: TICK_COL, size: 10 },
      },
    },
    hoverinfo: 'skip',   // next-step candidates sit exactly on cloud points
  };

  let maxT = 0;
  for (let i = 0; i < n; i++) if (data.t[i] > maxT) maxT = data.t[i];
  blochWalk.seqLen = maxT;
  blochWalk.meanPurity = null;
  if (data.mode === 'sample') {
    const sum = new Float64Array(maxT + 1), count = new Float64Array(maxT + 1);
    for (let i = 0; i < n; i++) {
      sum[data.t[i]] += Math.hypot(data.x[i], data.z[i]);
      count[data.t[i]]++;
    }
    blochWalk.meanPurity = Array.from(sum, (v, t) => v / count[t]);
  }

  blochReplay();
  drawBloch();
}

// Re-walk the same context under the current model (β may have changed).
function blochReplay() {
  const { model } = blochWalk;
  const path = [[1, 0, 0]];
  const kept = [];
  for (const tok of blochWalk.tokens) {
    const cur = path[path.length - 1];
    if (blochProbs(model, cur)[tok] < 1e-12) break;
    path.push(blochStep(model, cur, tok));
    kept.push(tok);
  }
  blochWalk.tokens = kept;
  blochWalk.path = path;
}

function blochStepToken(tok) {
  const { model, path } = blochWalk;
  if (currentProcess !== 'bloch' || !model) return;
  const cur = path[path.length - 1];
  if (blochProbs(model, cur)[tok] < 1e-12) return;
  blochWalk.tokens.push(tok);
  path.push(blochStep(model, cur, tok));
  drawBloch();
}

function blochRandomStep() {
  const { model, path } = blochWalk;
  if (currentProcess !== 'bloch' || !model) return;
  blochStepToken(sampleCategorical(blochProbs(model, path[path.length - 1]), Math.random));
  if (blochWalk.tokens.length >= BLOCH_AUTOPLAY_MAX) blochSetPlaying(false);
}

function blochUndo() {
  if (!blochWalk.tokens.length) return;
  blochWalk.tokens.pop();
  blochWalk.path.pop();
  drawBloch();
}

function blochReset() {
  blochSetPlaying(false);
  blochWalk.tokens = [];
  blochWalk.path = [[1, 0, 0]];
  drawBloch();
}

// β → 1/β keeps s and negates c: the same walk with the mirror toggled.
function blochInvertBeta() {
  const slider = $('slider-beta');
  if (!slider) return;
  slider.value = 1 / parseFloat(slider.value);
  slider.dispatchEvent(new Event('input'));
}

function blochSetPlaying(on) {
  clearInterval(blochWalk.timer);
  blochWalk.timer = on ? setInterval(blochRandomStep, BLOCH_PLAY_MS) : null;
  if (!DOM?.blochPlay) return;
  DOM.blochPlay.textContent = on ? 'Pause' : 'Play';
  DOM.blochPlay.classList.toggle('active', on);
}

function drawBloch() {
  const { model, path } = blochWalk;
  const cur   = path[path.length - 1];
  const probs = blochProbs(model, cur);
  plotBlochDisk(cur, probs);
  plotBlochPurity();
  updateBlochPanel(cur, probs);
}

function plotBlochDisk(cur, probs) {
  const { model, path } = blochWalk;

  // Where each token would take the belief next; marker area ∝ P(token).
  const next = { x: [], y: [], tok: [], p: [] };
  const fan  = { x: [], y: [] };
  for (let tok = 0; tok < 4; tok++) {
    if (probs[tok] < 1e-12) continue;
    const b = blochStep(model, cur, tok);
    next.x.push(b[1]); next.y.push(b[2]); next.tok.push(tok); next.p.push(probs[tok]);
    fan.x.push(cur[1], b[1], null); fan.y.push(cur[2], b[2], null);
  }
  const nextColors = next.tok.map(tok => BLOCH_TOKEN_COLORS[tok]);

  const fanTrace = {
    ...fan, mode: 'lines', type: 'scatter',
    line: { color: 'rgba(205, 197, 180, 0.35)', width: 1, dash: 'dot' },
    hoverinfo: 'skip',
  };
  const trail = path.slice(-BLOCH_TRAIL_MAX);
  const trailTrace = {
    x: trail.map(b => b[1]), y: trail.map(b => b[2]),
    mode: 'lines', type: 'scatter',
    line: { color: 'rgba(232, 224, 212, 0.55)', width: 1.3 },
    hoverinfo: 'skip',
  };
  const nextTrace = {
    x: next.x, y: next.y, name: 'next',
    mode: 'markers+text', type: 'scatter',
    text: next.tok.map(String), textposition: 'middle center',
    textfont: { size: 11, color: nextColors, family: 'Fira Code, monospace' },
    marker: {
      size: next.p.map(p => 40 * Math.sqrt(p)),
      color: 'rgba(8, 9, 12, 0.8)',
      line: { color: nextColors, width: 1.8 },
    },
    customdata: next.tok.map((tok, i) => [tok, next.p[i]]),
    hovertemplate: 'token %{customdata[0]}  ·  P = %{customdata[1]:.1%}<br>click to step<extra></extra>',
  };
  const curTrace = {
    x: [cur[1]], y: [cur[2]],
    mode: 'markers', type: 'scatter',
    marker: { size: 11, color: INK, line: { color: BG, width: 2 } },
    hovertemplate: 'belief<br>x = %{x:.3f}<br>z = %{y:.3f}<extra></extra>',
  };

  const axisLine = { type: 'line', layer: 'below', line: { color: '#1e2029', width: 1 } };
  const pole = (tok, x, y, label) => ({
    x, y, text: `<b>${tok}</b> ${label}`, showarrow: false,
    font: { size: 12, color: BLOCH_TOKEN_COLORS[tok], family: 'Fira Code, monospace' },
  });

  const layout = {
    xaxis: { visible: false, range: [-1.25, 1.25], scaleanchor: 'y' },
    yaxis: { visible: false, range: [-1.25, 1.25] },
    shapes: [
      { type: 'circle', layer: 'below', x0: -1, y0: -1, x1: 1, y1: 1,
        line: { color: '#2e3040', width: 1.5 } },
      { ...axisLine, x0: -1, x1: 1, y0: 0, y1: 0 },
      { ...axisLine, x0: 0, x1: 0, y0: -1, y1: 1 },
    ],
    annotations: [
      pole(0, 0, 1.1, '+z'), pole(1, 0, -1.1, '−z'),
      pole(2, 1.13, 0, '+x'), pole(3, -1.13, 0, '−x'),
    ],
    paper_bgcolor: BG, plot_bgcolor: BG,
    margin: { t: 16, b: 10, l: 10, r: 10 },
    showlegend: false,
    autosize: true,
  };

  Plotly.react('bloch-plot', [blochWalk.cloudTrace, fanTrace, trailTrace, nextTrace, curTrace],
               layout, plotConfig());
}

function plotBlochPurity() {
  const { path, meanPurity, seqLen: maxT } = blochWalk;
  const traces = [];
  if (meanPurity) {
    traces.push({
      x: meanPurity.map((_, t) => t), y: meanPurity,
      mode: 'lines', type: 'scatter',
      line: { color: '#5b9a8b', width: 2 },
      hovertemplate: 't = %{x}<br>⟨|r|⟩ = %{y:.3f}<extra>sample mean</extra>',
    });
  }
  traces.push({
    x: path.map((_, t) => t), y: path.map(blochRadius),
    mode: 'lines+markers', type: 'scatter',
    line: { color: ACCENT, width: 1.5 }, marker: { size: 4, color: ACCENT },
    hovertemplate: 't = %{x}<br>|r| = %{y:.3f}<extra>your walk</extra>',
  });

  const layout = {
    title: {
      text: `purity |r| vs context length  ·  <span style="color:#5b9a8b">sample mean</span>  <span style="color:${ACCENT}">your walk</span>`,
      font: { size: 11, color: TICK_COL }, x: 0.01, xanchor: 'left',
    },
    xaxis: { ...axisStyle(), range: [0, Math.max(maxT, path.length - 1, 10)] },
    yaxis: { ...axisStyle(), range: [0, 1.04], tickvals: [0, 0.5, 1] },
    paper_bgcolor: BG, plot_bgcolor: PLOT_BG,
    margin: { t: 30, b: 24, l: 34, r: 12 },
    showlegend: false, autosize: true,
  };
  Plotly.react('bloch-purity-plot', traces, layout, plotConfig());
}

function updateBlochPanel(cur, probs) {
  const { model, tokens } = blochWalk;

  DOM.blochDpad.querySelectorAll('[data-tok]').forEach(btn => {
    const p = probs[parseInt(btn.dataset.tok)];
    btn.style.setProperty('--p', Math.min(1, p / 0.5));
    btn.querySelector('.dpad-p').textContent = `${(p * 100).toFixed(1)}%`;
    btn.disabled = p < 1e-12;
  });

  DOM.blochXZ.textContent = `(${fmtSigned(cur[1])}, ${fmtSigned(cur[2])})`;
  DOM.blochR.textContent  = fmtFixed(blochRadius(cur), 4);

  const SHOWN = 18;
  const recent = tokens.slice(-SHOWN)
    .map(tok => `<span style="color:${BLOCH_TOKEN_COLORS[tok]}">${tok}</span>`).join('');
  DOM.blochTokens.innerHTML = tokens.length
    ? (tokens.length > SHOWN ? '…' : '') + recent
    : '∅';

  DOM.blochSC.innerHTML = `s = ${fmtSigned(model.s)} &nbsp; c = ${fmtSigned(model.c)}`;
  DOM.blochFlipNote.textContent = model.c < 0
    ? 'β > 1: each step also flips the sign of the other coordinate.'
    : 'β < 1: no flip, a plain N/S/E/W walk.';
}

// ── Number formatting ─────────────────────────────────────────────────────────
function fmtFixed(v, d = 3) {
  if (v === Infinity) return '∞';
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

function fmtSigned(v, d = 3) {
  if (!Number.isFinite(v)) return '—';
  const s = v.toFixed(d);
  return s.startsWith('-') ? '−' + s.slice(1) : s;
}

function fmtComplex(z, d = 3) {
  if (Math.abs(z.im) < 1e-10) return fmtSigned(z.re, d);
  const sign = z.im >= 0 ? '+' : '−';
  return `${fmtSigned(z.re, d)} ${sign} ${Math.abs(z.im).toFixed(d)}i`;
}
