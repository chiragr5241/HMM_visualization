/* ═══════════════════════════════════════════════════════════════════
   Belief State Geometry — computation engine (pure JS, no backend)
   Ports all Python/JAX computation to client-side JavaScript.
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ── Linear algebra utilities ──────────────────────────────────────────────────
// All matrices are at most 4×4, so no library needed.

function vecMatMul(vec, mat) {
  // (N,) @ (N,N) → (N,)  — row vector times matrix
  const n = vec.length;
  const out = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += vec[i] * mat[i][j];
    out[j] = s;
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function vecSum(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return s;
}

function outerProduct(a, b) {
  const n = a.length, m = b.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    out[i] = new Float64Array(m);
    for (let j = 0; j < m; j++) out[i][j] = a[i] * b[j];
  }
  return out;
}

function matScale(mat, s) {
  return mat.map(row => row.map(v => v * s));
}

function matAdd(a, b) {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

function logsumexp(logVals) {
  let maxVal = -Infinity;
  for (let i = 0; i < logVals.length; i++) if (logVals[i] > maxVal) maxVal = logVals[i];
  if (maxVal === -Infinity) return -Infinity;
  let s = 0;
  for (let i = 0; i < logVals.length; i++) s += Math.exp(logVals[i] - maxVal);
  return maxVal + Math.log(s);
}

// ── Seedable PRNG (mulberry32) ─────────────────────────────────────────────
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function sampleCategorical(probs, rng) {
  const u = rng();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (u < cum) return i;
  }
  return probs.length - 1;
}

// ── Eigenvalue computation ─────────────────────────────────────────────────

function stationaryDistribution(stateTransitionMatrix) {
  // Left eigenvector of stateTransitionMatrix with eigenvalue 1.
  // Equivalent to: right eigenvector of stateTransitionMatrix^T with eigenvalue 1.
  // Use power iteration on the transpose.
  const n = stateTransitionMatrix.length;
  let v = new Float64Array(n).fill(1 / n);
  for (let iter = 0; iter < 2000; iter++) {
    // v_new = v @ stateTransitionMatrix (left multiply)
    const vNew = vecMatMul(v, stateTransitionMatrix);
    const s = vecSum(vNew);
    if (s === 0) break;
    for (let i = 0; i < n; i++) vNew[i] /= s;
    // Check convergence
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(vNew[i] - v[i]);
    v = vNew;
    if (diff < 1e-14) break;
  }
  // Normalize to sum to 1
  const s = vecSum(v);
  for (let i = 0; i < n; i++) v[i] /= s;
  return v;
}

function dominantEig(mat) {
  // Find dominant eigenvalue and RIGHT eigenvector via power iteration.
  const n = mat.length;
  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = 1.0;
  let eigenvalue = 1;
  for (let iter = 0; iter < 2000; iter++) {
    // v_new = mat @ v (right multiply)
    const vNew = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += mat[i][j] * v[j];
      vNew[i] = s;
    }
    // Find max abs for normalization
    let maxAbs = 0;
    for (let i = 0; i < n; i++) if (Math.abs(vNew[i]) > maxAbs) maxAbs = Math.abs(vNew[i]);
    if (maxAbs === 0) break;
    eigenvalue = maxAbs;
    for (let i = 0; i < n; i++) vNew[i] /= maxAbs;
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(vNew[i] - v[i]);
    v = vNew;
    if (diff < 1e-14) break;
  }
  return { eigenvector: v, eigenvalue };
}

// ── PCA via covariance matrix ──────────────────────────────────────────────

function pca2D(beliefStates) {
  const n = beliefStates.length;
  if (n === 0) return { x: [], y: [], varRatios: [0, 0] };
  const d = beliefStates[0].length;

  // Mean
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) mean[j] += beliefStates[i][j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  // Centered data
  const Xc = beliefStates.map(row => {
    const r = new Float64Array(d);
    for (let j = 0; j < d; j++) r[j] = row[j] - mean[j];
    return r;
  });

  // Covariance C = Xc^T @ Xc (d×d)
  const C = [];
  for (let i = 0; i < d; i++) {
    C[i] = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Xc[k][i] * Xc[k][j];
      C[i][j] = s;
    }
  }

  // Total variance
  let totalVar = 0;
  for (let i = 0; i < d; i++) totalVar += C[i][i];

  // First eigenvector via power iteration
  const { eigenvector: v1, eigenvalue: ev1 } = dominantEig(C);

  // Deflate: C' = C - ev1 * outer(v1, v1)
  const C2 = [];
  for (let i = 0; i < d; i++) {
    C2[i] = new Float64Array(d);
    for (let j = 0; j < d; j++) C2[i][j] = C[i][j] - ev1 * v1[i] * v1[j];
  }

  // Second eigenvector
  const { eigenvector: v2, eigenvalue: ev2 } = dominantEig(C2);

  // Project
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = dot(Xc[i], v1);
    y[i] = dot(Xc[i], v2);
  }

  const varRatios = totalVar > 0 ? [ev1 / totalVar, ev2 / totalVar] : [0, 0];
  return { x: Array.from(x), y: Array.from(y), varRatios };
}

// ── Transition matrix builders ─────────────────────────────────────────────

function buildMess3(x, a) {
  const b = (1 - a) / 2;
  const y = 1 - 2 * x;
  const ay = a * y, bx = b * x, by = b * y, ax = a * x;
  return [
    [[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
    [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
    [[by, bx, ax], [bx, by, ax], [bx, bx, ay]],
  ];
}

function buildMess3_2(x, a, p, q, r) {
  const base = buildMess3(x, a);
  const T = [
    base[0].map((row, i) => row.map((v, j) => p * base[0][i][j] + q * base[1][i][j] + r * base[2][i][j])),
    base[0].map((row, i) => row.map((v, j) => (1 - p) * base[0][i][j] + (1 - q) * base[1][i][j] + (1 - r) * base[2][i][j])),
  ];
  return T;
}

function buildRiver() {
  return [
    [[0.05, 0.0, 0.0], [0.1, 0.1, 0.2], [0.0, 0.5, 0.0]],
    [[0.85, 0.0, 0.1], [0.1, 0.0, 0.5], [0.0, 0.45, 0.05]],
  ];
}

function buildLeopard(x) {
  return [
    [[0.0, 0.0, 0.3465], [0.6435, 0.0, 0.0], [0.0, 0.99 * x, 0.0]],
    [[0.005, 0.005, 0.6435], [0.3465, 0.005, 0.005], [0.005, 0.99 * (1 - x), 0.005]],
  ];
}

function buildFern(x) {
  return [
    [[0.3942, 0.00512, 0.0381], [0.0, 0.53, 0.0], [0.0, 0.326 * x, 0.554]],
    [[0.3358, 0.01088, 0.2159], [0.0, 0.0, 0.47], [0.12, 0.326 * (1 - x), 0.0]],
  ];
}

function buildFanizza(alpha, lamb) {
  const cosA = Math.cos(alpha), sinA = Math.sin(alpha);
  const denom = 1 - 2 * lamb * cosA + lamb * lamb;
  const a_la = (1 - lamb * cosA + lamb * sinA) / denom;
  const b_la = (1 - lamb * cosA - lamb * sinA) / denom;

  const pi0 = [
    1 - (2 / (1 - lamb) - a_la - b_la) / 4,
    1 / (2 * (1 - lamb)),
    -a_la / 4,
    -b_la / 4,
  ];
  const w = [
    1,
    1 - lamb,
    1 + lamb * (sinA - cosA),
    1 - lamb * (sinA + cosA),
  ];

  const Da = outerProduct(w, pi0);
  const Db = [
    [0, 0, 0, 0],
    [0, lamb, 0, 0],
    [0, 0, lamb * cosA, -lamb * sinA],
    [0, 0, lamb * sinA, lamb * cosA],
  ];

  return { T: [Da, Db], isGHMM: true };
}

// ── Model initialization ───────────────────────────────────────────────────

function sumTransitionMatrices(T) {
  const numSym = T.length, n = T[0].length;
  const S = [];
  for (let i = 0; i < n; i++) {
    S[i] = new Float64Array(n);
    for (let s = 0; s < numSym; s++)
      for (let j = 0; j < n; j++) S[i][j] += T[s][i][j];
  }
  return S;
}

function initializeModel(T, isGHMM) {
  const numStates = T[0].length;
  const numSymbols = T.length;
  const stm = sumTransitionMatrices(T);

  if (isGHMM) {
    // Find dominant eigenvalue and right eigenvector
    const { eigenvector: rawEigvec, eigenvalue: principalEv } = dominantEig(stm);

    // Scale T so principal eigenvalue = 1
    let scaledT = T;
    if (Math.abs(principalEv - 1) > 1e-10) {
      scaledT = T.map(m => m.map(row => row.map(v => v / principalEv)));
    }

    // Normalize eigenvector: sum to numStates (matches Python)
    const eigSum = vecSum(rawEigvec);
    const normEigvec = new Float64Array(numStates);
    for (let i = 0; i < numStates; i++) normEigvec[i] = rawEigvec[i] / eigSum * numStates;

    // Stationary distribution (left eigenvector of stm^T → which is left eigenvector of stm)
    const scaledStm = sumTransitionMatrices(scaledT);
    const initialState = stationaryDistribution(scaledStm);

    const normConst = dot(initialState, normEigvec);

    return { T: scaledT, initialState, normEigvec, normConst, numStates, numSymbols, isGHMM: true };
  }

  // HMM: check if principal eigenvalue is 1, scale if not
  const { eigenvalue: principalEv } = dominantEig(stm);
  let scaledT = T;
  if (Math.abs(principalEv - 1) > 1e-10) {
    scaledT = T.map(m => m.map(row => row.map(v => v / principalEv)));
  }

  const scaledStm = sumTransitionMatrices(scaledT);
  const initialState = stationaryDistribution(scaledStm);

  // For HMM, normalizing eigenvector is all-ones
  const normEigvec = new Float64Array(numStates).fill(1);
  const normConst = vecSum(initialState);

  return { T: scaledT, initialState, normEigvec, normConst, numStates, numSymbols, isGHMM: false };
}

// ── Build model from request ──────────────────────────────────────────────

function buildModel(process, params) {
  const p = params;
  switch (process) {
    case 'mess3':
      return initializeModel(buildMess3(p.x ?? 0.15, p.a ?? 0.6), false);
    case 'mess3_2':
      return initializeModel(buildMess3_2(p.x ?? 0.15, p.a ?? 0.6, p.p ?? 0.7, p.q ?? 0.3, p.r ?? 0.5), false);
    case 'river':
      return initializeModel(buildRiver(), false);
    case 'leopard':
      return initializeModel(buildLeopard(p.x ?? 0.5), false);
    case 'fern':
      return initializeModel(buildFern(p.x ?? 0.5), false);
    case 'fanizza': {
      const { T, isGHMM } = buildFanizza(p.alpha ?? 2000.0, p.lamb ?? 0.49);
      return initializeModel(T, isGHMM);
    }
    default:
      throw new Error(`Unknown process: ${process}`);
  }
}

// ── Belief state sampling ──────────────────────────────────────────────────

function sampleBeliefStates(model, batchSize, seqLen, seed) {
  const rng = mulberry32(seed);
  const { T, initialState, normEigvec, numStates, numSymbols, isGHMM } = model;
  const allStates = [];

  for (let b = 0; b < batchSize; b++) {
    let state = Float64Array.from(initialState);
    allStates.push(Float64Array.from(state));

    for (let t = 0; t < seqLen; t++) {
      // Compute observation probabilities
      const pObs = new Float64Array(numSymbols);
      if (isGHMM) {
        const denominator = dot(state, normEigvec);
        for (let s = 0; s < numSymbols; s++) {
          const stateAfterObs = vecMatMul(state, T[s]);
          pObs[s] = dot(stateAfterObs, normEigvec) / denominator;
        }
      } else {
        for (let s = 0; s < numSymbols; s++) {
          const stateAfterObs = vecMatMul(state, T[s]);
          pObs[s] = vecSum(stateAfterObs);
        }
      }

      // Normalize (handle numerical imprecision)
      const pSum = vecSum(pObs);
      for (let s = 0; s < numSymbols; s++) pObs[s] /= pSum;

      // Sample observation
      const obs = sampleCategorical(pObs, rng);

      // Update belief state
      const unnorm = vecMatMul(state, T[obs]);
      if (isGHMM) {
        const norm = dot(unnorm, normEigvec);
        for (let i = 0; i < numStates; i++) state[i] = unnorm[i] / norm;
      } else {
        const norm = vecSum(unnorm);
        for (let i = 0; i < numStates; i++) state[i] = unnorm[i] / norm;
      }

      allStates.push(Float64Array.from(state));
    }
  }

  return allStates;
}

// ── MSP tree enumeration ──────────────────────────────────────────────────

function enumerateMSP_HMM(model, maxSeqLen) {
  // Log-space BFS for numerical stability (handles zeros in RIVER)
  const { T, initialState, numStates, numSymbols } = model;

  // Precompute log transition matrices
  const logT = T.map(m => m.map(row => row.map(v => Math.log(v))));

  const logInitial = new Float64Array(numStates);
  for (let i = 0; i < numStates; i++) logInitial[i] = Math.log(initialState[i]);

  const results = [];

  // BFS queue: { logUnnormBelief, depth }
  const queue = [{ logUnnormBelief: logInitial, depth: 0 }];

  while (queue.length > 0) {
    const node = queue.shift();

    // Normalize to get belief state
    const lse = logsumexp(node.logUnnormBelief);
    const logBelief = new Float64Array(numStates);
    for (let i = 0; i < numStates; i++) logBelief[i] = node.logUnnormBelief[i] - lse;
    const belief = new Float64Array(numStates);
    for (let i = 0; i < numStates; i++) belief[i] = Math.exp(logBelief[i]);

    results.push(belief);

    if (node.depth < maxSeqLen) {
      for (let obs = 0; obs < numSymbols; obs++) {
        // log(unnorm_child) = logsumexp_over_i(logUnnormParent[i] + logT[obs][i][j]) for each j
        const childLogUnnorm = new Float64Array(numStates);
        for (let j = 0; j < numStates; j++) {
          const vals = new Float64Array(numStates);
          for (let i = 0; i < numStates; i++) vals[i] = node.logUnnormBelief[i] + logT[obs][i][j];
          childLogUnnorm[j] = logsumexp(vals);
        }
        queue.push({ logUnnormBelief: childLogUnnorm, depth: node.depth + 1 });
      }
    }
  }

  return results;
}

function enumerateMSP_GHMM(model, maxSeqLen, probThreshold) {
  const { T, initialState, normEigvec, normConst, numStates, numSymbols } = model;

  const results = [];

  // BFS queue: { unnormBelief, prob, depth }
  const queue = [{ unnormBelief: Float64Array.from(initialState), depth: 0 }];

  while (queue.length > 0) {
    const node = queue.shift();

    // Normalize
    const normFactor = dot(node.unnormBelief, normEigvec);
    const belief = new Float64Array(numStates);
    for (let i = 0; i < numStates; i++) belief[i] = node.unnormBelief[i] / normFactor;
    const prob = normFactor / normConst;

    if (prob < (probThreshold || 0)) continue;

    results.push(belief);

    if (node.depth < maxSeqLen) {
      for (let obs = 0; obs < numSymbols; obs++) {
        const childUnnorm = vecMatMul(node.unnormBelief, T[obs]);
        queue.push({ unnormBelief: childUnnorm, depth: node.depth + 1 });
      }
    }
  }

  return results;
}

// ── Projection & coloring ──────────────────────────────────────────────────

const SQRT3_HALF = Math.sqrt(3) / 2;

function barycentric(bs) {
  return { x: bs[1] + 0.5 * bs[2], y: SQRT3_HALF * bs[2] };
}

function beliefToRGB(bs) {
  return {
    r: Math.round(Math.min(255, Math.max(0, bs[0] * 255))),
    g: Math.round(Math.min(255, Math.max(0, bs[1] * 255))),
    b: Math.round(Math.min(255, Math.max(0, bs[2] * 255))),
  };
}

function ghmmObsProbDist(belief, model) {
  // P(obs=sym) for each symbol
  const { T, normEigvec, numSymbols } = model;
  const denominator = dot(belief, normEigvec);
  const pObs = new Float64Array(numSymbols);
  for (let s = 0; s < numSymbols; s++) {
    const stateAfterObs = vecMatMul(belief, T[s]);
    pObs[s] = dot(stateAfterObs, normEigvec) / denominator;
  }
  return pObs;
}

// ── Format results ─────────────────────────────────────────────────────────

function formatSimplex(beliefStates) {
  const n = beliefStates.length;
  const xArr = new Float64Array(n), yArr = new Float64Array(n);
  const rArr = new Uint8Array(n), gArr = new Uint8Array(n), bArr = new Uint8Array(n);
  const bsArr = [];

  for (let i = 0; i < n; i++) {
    const bs = beliefStates[i];
    const { x, y } = barycentric(bs);
    xArr[i] = x;
    yArr[i] = y;
    const rgb = beliefToRGB(bs);
    rArr[i] = rgb.r;
    gArr[i] = rgb.g;
    bArr[i] = rgb.b;
    bsArr.push([
      Math.round(bs[0] * 100000) / 100000,
      Math.round(bs[1] * 100000) / 100000,
      Math.round(bs[2] * 100000) / 100000,
    ]);
  }

  return {
    type: 'simplex',
    x: Array.from(xArr), y: Array.from(yArr),
    r: Array.from(rArr), g: Array.from(gArr), b: Array.from(bArr),
    bs: bsArr, n,
  };
}

function formatPCA(beliefStates, model) {
  const n = beliefStates.length;
  const colorVal = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const pObs = ghmmObsProbDist(beliefStates[i], model);
    colorVal[i] = pObs[0];
  }

  const { x, y, varRatios } = pca2D(beliefStates);

  return {
    type: 'pca',
    x, y,
    color_val: Array.from(colorVal),
    n,
    var_ratios: varRatios,
  };
}

// ── Top-level compute dispatcher ──────────────────────────────────────────

function computeResult(process, params, mode, modeParams) {
  const model = buildModel(process, params);
  const isGHMM = model.isGHMM;

  let beliefStates;
  if (mode === 'sample') {
    beliefStates = sampleBeliefStates(
      model,
      modeParams.batchSize || 500,
      modeParams.seqLen || 20,
      modeParams.seed || 0,
    );
  } else {
    // enumerate
    if (isGHMM) {
      beliefStates = enumerateMSP_GHMM(model, modeParams.maxSeqLen || 10, 1e-9);
    } else {
      beliefStates = enumerateMSP_HMM(model, modeParams.maxSeqLen || 10);
    }
  }

  let result;
  if (isGHMM) {
    result = formatPCA(beliefStates, model);
  } else {
    result = formatSimplex(beliefStates);
  }
  result.mode = mode;
  return result;
}
