/* eslint-disable no-undef */
/**
 * Loudness Meter (EBU R128 / ITU-R BS.1770 approx) – client-only
 * - File analysis via OfflineAudioContext → K-weighting (HP ~60 Hz, HS ~4 kHz +4 dB)
 * - Series:
 *    Momentary (M) 400 ms, hop 100 ms
 *    Short-term (S) 3 s, hop 100 ms
 * - Gating simplified:
 *    Integrated: absolute gate −70 LUFS, then relative gate −10 LU (1 iter)
 *    LRA: percentiles P10–P95 over S with relative gate (LUFS-I − 20)
 * - True Peak (dBTP): oversampling 4× via Catmull-Rom cubic interpolation; peak over channels
 * - PLR = dBTP − LUFS-I
 * - DR≈ = P95(S) − P5(S)
 * - UI: metric cards + two Canvas charts
 * - Live (optional): getUserMedia + Biquads + AudioWorklet for stable timing
 */

const els = {
  fileInput: document.getElementById('fileInput'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  liveBtn: document.getElementById('liveBtn'),
  darkToggle: document.getElementById('darkToggle'),
  lufsI: document.getElementById('lufsI'),
  lufsS: document.getElementById('lufsS'),
  lufsSmax: document.getElementById('lufsSmax'),
  tSmax: document.getElementById('tSmax'),
  lufsM: document.getElementById('lufsM'),
  lufsMmax: document.getElementById('lufsMmax'),
  tMmax: document.getElementById('tMmax'),
  lra: document.getElementById('lra'),
  dbtp: document.getElementById('dbtp'),
  plr: document.getElementById('plr'),
  dr: document.getElementById('dr'),
  canvasM: document.getElementById('canvasM'),
  canvasS: document.getElementById('canvasS'),
  log: document.getElementById('log'),
};

const M_WINDOW_SEC = 0.400;
const S_WINDOW_SEC = 3.000;
const DEFAULT_HOP_SEC = 0.100;
const K_OFFSET_DB = -0.691; // ITU-R BS.1770 reference offset (approx) for LKFS/LUFS

let audioCtx = null;
let live = {
  ctx: null,
  stream: null,
  workletNode: null,
  hp: null,
  hs: null,
  running: false,
};

let worker = null;

// ---------- Utilities ----------

function logln(...args) {
  console.log(...args);
  els.log.textContent += args.map(a => String(a)).join(' ') + '\n';
  els.log.scrollTop = els.log.scrollHeight;
}

function fmtDb(x) {
  if (!isFinite(x)) return '−∞';
  return (Math.round(x * 10) / 10).toFixed(1);
}
function fmtLU(x) {
  if (!isFinite(x)) return '−∞';
  return (Math.round(x * 10) / 10).toFixed(1);
}
function fmtTime(seconds) {
  if (!isFinite(seconds)) return '—';
  const s = Math.max(0, seconds);
  const mm = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  const msec = Math.round((s - Math.floor(s)) * 1000);
  return `${mm.toString().padStart(2,'0')}:${ss.toString().padStart(2,'0')}.${msec.toString().padStart(3,'0')}`;
}

function maxWithIndex(arr) {
  if (!arr.length) return { max: -Infinity, idx: -1 };
  let m = arr[0], i = 0;
  for (let k = 1; k < arr.length; k++) {
    if (arr[k] > m) { m = arr[k]; i = k; }
  }
  return { max: m, idx: i };
}

function percentile(sortedArr, p /* 0..100 */) {
  if (!sortedArr.length) return NaN;
  if (p <= 0) return sortedArr[0];
  if (p >= 100) return sortedArr[sortedArr.length - 1];
  const pos = (p / 100) * (sortedArr.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const frac = pos - lo;
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * frac;
}

// Simple line chart
function drawSeries(canvas, times, values, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // padding
  const pL = 40, pR = 10, pT = 16, pB = 24;
  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(0, 0, W, H);

  if (!values?.length || !times?.length) {
    ctx.fillStyle = '#888';
    ctx.fillText('Sin datos', pL, H/2);
    return;
  }

  const t0 = times[0];
  const t1 = times[times.length - 1];
  const vMin = Math.min(...values.filter(Number.isFinite));
  const vMax = Math.max(...values.filter(Number.isFinite));
  const minY = Math.floor((Math.min(vMin, -70) - 1) / 5) * 5; // grid to −∞..-70
  const maxY = Math.ceil((vMax + 1) / 5) * 5;

  const X = t => pL + ((t - t0) / (t1 - t0 || 1)) * (W - pL - pR);
  const Y = v => pT + (1 - (v - minY) / (maxY - minY || 1)) * (H - pT - pB);

  // grid
  const ctxGrid = ctx;
  ctxGrid.strokeStyle = 'rgba(128,128,128,0.2)';
  ctxGrid.lineWidth = 1;
  ctxGrid.beginPath();
  for (let y = Math.ceil(minY/5)*5; y <= maxY; y += 5) {
    const yy = Y(y);
    ctxGrid.moveTo(pL, yy);
    ctxGrid.lineTo(W - pR, yy);
  }
  ctxGrid.stroke();

  // axes labels (min/max)
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted') || '#aaa';
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${maxY} LUFS`, pL - 6, Y(maxY));
  ctx.fillText(`${minY} LUFS`, pL - 6, Y(minY));

  // series
  ctx.lineWidth = 2;
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent-strong') || '#2d8fe2';
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const xx = X(times[i]);
    const yy = Y(values[i]);
    if (i === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  }
  ctx.stroke();

  // max marker
  const { idx } = maxWithIndex(values);
  if (idx >= 0) {
    const xx = X(times[idx]);
    const yy = Y(values[idx]);
    ctx.fillStyle = '#ff6b6b';
    ctx.beginPath();
    ctx.arc(xx, yy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---------- Web Worker (compute engine) ----------

function ensureWorker() {
  if (worker) return worker;
  const code = `
    const K_OFFSET_DB = -0.691;
    function lufsFromMS(ms) {
      if (ms <= 0) return Number.NEGATIVE_INFINITY;
      return 10 * Math.log10(ms) + K_OFFSET_DB;
    }
    function msFromLUFS(lufs) {
      return Math.pow(10, (lufs - K_OFFSET_DB) / 10);
    }
    function maxWithIndex(arr) {
      if (!arr.length) return { max: -Infinity, idx: -1 };
      let m = arr[0], i = 0;
      for (let k = 1; k < arr.length; k++) { if (arr[k] > m) { m = arr[k]; i = k; } }
      return { max: m, idx: i };
    }
    function percentile(sortedArr, p) {
      if (!sortedArr.length) return NaN;
      if (p <= 0) return sortedArr[0];
      if (p >= 100) return sortedArr[sortedArr.length - 1];
      const pos = (p / 100) * (sortedArr.length - 1);
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      const frac = pos - lo;
      return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * frac;
    }
    // Compute windowed mean squares for multi-channel using prefix sums (O(N))
    function seriesLUFS(channels, sampleRate, winSec, hopSec) {
      const win = Math.max(1, Math.floor(winSec * sampleRate));
      const hop = Math.max(1, Math.floor(hopSec * sampleRate));
      const length = channels[0].length;
      const nCh = channels.length;

      // Prefix sums of squares per channel
      const P = new Array(nCh);
      for (let c = 0; c < nCh; c++) {
        const x = channels[c];
        const Pc = new Float64Array(length + 1);
        let acc = 0;
        Pc[0] = 0;
        for (let i = 0; i < length; i++) { acc += x[i]*x[i]; Pc[i+1] = acc; }
        P[c] = Pc;
      }

      const lufs = [];
      const msBlocks = [];
      const times = [];
      for (let s = 0; s + win <= length; s += hop) {
        let sumMS = 0;
        for (let c = 0; c < nCh; c++) {
          const Pc = P[c];
          const sumsq = Pc[s + win] - Pc[s];
          const ms_c = sumsq / win;
          sumMS += ms_c; // BS.1770: suma de energías por canal (peso 1 para L/R)
        }
        msBlocks.push(sumMS);
        lufs.push(lufsFromMS(sumMS));
        times.push((s + win * 0.5) / sampleRate);
      }
      return { lufs, msBlocks, times, win, hop };
    }

    // Integrated gating (abs −70, then rel −10 LU)
    function integratedLUFSFromS(msBlocks, lufsBlocks) {
      if (!msBlocks.length) return { lufsI: Number.NEGATIVE_INFINITY, count: 0, gateRel: Number.NEGATIVE_INFINITY };
      const ABS_GATE = -70;
      const keptAbs = [];
      for (let i = 0; i < lufsBlocks.length; i++) {
        if (lufsBlocks[i] > ABS_GATE) keptAbs.push(msBlocks[i]);
      }
      if (!keptAbs.length) return { lufsI: Number.NEGATIVE_INFINITY, count: 0, gateRel: Number.NEGATIVE_INFINITY };
      const meanAbs = keptAbs.reduce((a,b)=>a+b,0) / keptAbs.length;
      const prelim = 10 * Math.log10(meanAbs) + K_OFFSET_DB;
      const relGate = prelim - 10;
      const keptRel = [];
      for (let i = 0; i < lufsBlocks.length; i++) {
        if (lufsBlocks[i] >= relGate) keptRel.push(msBlocks[i]);
      }
      if (!keptRel.length) {
        return { lufsI: prelim, count: keptAbs.length, gateRel: relGate };
      }
      const meanRel = keptRel.reduce((a,b)=>a+b,0) / keptRel.length;
      const result = 10 * Math.log10(meanRel) + K_OFFSET_DB;
      return { lufsI: result, count: keptRel.length, gateRel: relGate };
    }

    // LRA: P10..P95 of S with gate relative −20 LU from integrated
    function computeLRA(lufsS, lufsI) {
      if (!lufsS.length || !isFinite(lufsI)) return { lra: 0, p10: NaN, p95: NaN };
      const thr = lufsI - 20;
      const gated = lufsS.filter(v => isFinite(v) && v >= thr);
      const base = gated.length ? gated : lufsS.filter(isFinite);
      const sorted = base.slice().sort((a,b)=>a-b);
      const p10 = percentile(sorted, 10);
      const p95 = percentile(sorted, 95);
      return { lra: p95 - p10, p10, p95, thr };
    }

    // DR approx: P95 - P5 over all S
    function computeDR(lufsS) {
      const arr = lufsS.filter(isFinite).slice().sort((a,b)=>a-b);
      if (!arr.length) return { dr: 0, p5: NaN, p95: NaN };
      const p5 = percentile(arr, 5);
      const p95 = percentile(arr, 95);
      return { dr: p95 - p5, p5, p95 };
    }

    // True Peak (4× oversampling with Catmull-Rom). Returns dBTP and linear peak
    function truePeakLinear(channels) {
      let peak = 0;
      for (let c = 0; c < channels.length; c++) {
        const x = channels[c];
        const N = x.length;
        if (!N) continue;
        // include original samples
        for (let i = 0; i < N; i++) {
          const abs = Math.abs(x[i]);
          if (abs > peak) peak = abs;
        }
        // cubic oversampling
        for (let i = 0; i < N - 1; i++) {
          const s0 = i > 0 ? x[i-1] : x[i];
          const s1 = x[i];
          const s2 = x[i+1];
          const s3 = (i+2 < N) ? x[i+2] : x[i+1];
          for (let k = 1; k < 4; k++) { // 4x: t = 0.25, 0.5, 0.75
            const t = k * 0.25;
            const a = 2*s1;
            const b = -s0 + s2;
            const c2 = 2*s0 - 5*s1 + 4*s2 - s3;
            const d = -s0 + 3*s1 - 3*s2 + s3;
            const y = 0.5 * (a + b*t + c2*t*t + d*t*t*t);
            const abs = Math.abs(y);
            if (abs > peak) peak = abs;
          }
        }
      }
      const dbtp = (peak > 0) ? 20*Math.log10(peak) : Number.NEGATIVE_INFINITY;
      return { peak, dbtp };
    }

    onmessage = async (ev) => {
      const { type, payload } = ev.data || {};
      if (type === 'compute') {
        const { channels, sampleRate, hopSec, mWindowSec, sWindowSec } = payload;
        try {
          const M = seriesLUFS(channels, sampleRate, mWindowSec, hopSec);
          const S = seriesLUFS(channels, sampleRate, sWindowSec, hopSec);

          const integ = integratedLUFSFromS(M.msBlocks, M.lufs);
          const lraObj = computeLRA(S.lufs, integ.lufsI);
          const drObj = computeDR(S.lufs);
          const tp = truePeakLinear(channels);

          const mMax = maxWithIndex(M.lufs);
          const sMax = maxWithIndex(S.lufs);

          postMessage({
            ok: true,
            result: {
              timesM: M.times, lufsM: M.lufs, timesS: S.times, lufsS: S.lufs,
              lufsI: integ.lufsI, gateRel: integ.gateRel,
              lra: lraObj.lra, lraThr: integ.lufsI - 20,
              dbtp: tp.dbtp,
              plr: tp.dbtp - integ.lufsI,
              dr: drObj.dr,
              mMaxIdx: mMax.idx,
              sMaxIdx: sMax.idx,
            }
          });
        } catch (e) {
          postMessage({ ok: false, error: e?.message || String(e) });
        }
      }
    };
  `;
  const blob = new Blob([code], { type: 'application/javascript' });
  worker = new Worker(URL.createObjectURL(blob));
  return worker;
}

// ---------- K-weighted render (Offline) ----------

async function renderKWeighted(buffer) {
  // Offline rendering: buffer -> [highpass ~60 Hz] -> [highshelf ~4 kHz, +4 dB] -> destination
  const sr = buffer.sampleRate;
  const ch = buffer.numberOfChannels;
  const ctx = new OfflineAudioContext(ch, buffer.length, sr);
  const src = ctx.createBufferSource();
  src.buffer = buffer;

  const hp = ctx.createBiquadFilter();
  hp.type = 'highpass';
  hp.frequency.value = 60;     // ~60 Hz
  hp.Q.value = Math.SQRT1_2;   // ~0.707

  const hs = ctx.createBiquadFilter();
  hs.type = 'highshelf';
  hs.frequency.value = 4000;   // ~4 kHz
  hs.gain.value = 4;           // +4 dB
  hs.Q.value = 0.707;

  src.connect(hp).connect(hs).connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered;
}

// ---------- File analysis pipeline ----------

async function analyzeFile(file) {
  if (!file) {
    alert('Selecciona un archivo de audio primero.');
    return;
  }
  logln('Decodificando:', file.name);
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  const abuf = await file.arrayBuffer();
  const buffer = await audioCtx.decodeAudioData(abuf);
  logln('Audio decodificado:', buffer.sampleRate + ' Hz, ' + buffer.numberOfChannels + ' ch, ' + (buffer.duration.toFixed(3)) + ' s');

  logln('Render K-weighted (offline)…');
  const kbuf = await renderKWeighted(buffer);

  // Collect channels data copies (transferable-friendly)
  const channels = [];
  for (let c = 0; c < kbuf.numberOfChannels; c++) {
    channels.push(kbuf.getChannelData(c).slice());
  }

  const w = ensureWorker();
  const result = await new Promise((resolve, reject) => {
    const onMsg = (ev) => {
      if (ev.data?.ok) { w.removeEventListener('message', onMsg); resolve(ev.data.result); }
      else if (ev.data?.ok === false) { w.removeEventListener('message', onMsg); reject(new Error(ev.data?.error || 'Worker error')); }
    };
    w.addEventListener('message', onMsg);
    w.postMessage({
      type: 'compute',
      payload: {
        channels,
        sampleRate: kbuf.sampleRate,
        hopSec: DEFAULT_HOP_SEC,
        mWindowSec: M_WINDOW_SEC,
        sWindowSec: S_WINDOW_SEC,
      }
    });
  });

  logln('Cálculo OK. Actualizando UI…');
  updateUIFromResult(result);
}

function updateUIFromResult(r) {
  // Metrics
  els.lufsI.textContent = fmtLU(r.lufsI);
  els.lufsS.textContent = (r.lufsS.length ? fmtLU(r.lufsS[r.lufsS.length - 1]) : '—');
  els.lufsM.textContent = (r.lufsM.length ? fmtLU(r.lufsM[r.lufsM.length - 1]) : '—');

  // Maxima & times
  if (r.mMaxIdx >= 0) {
    els.lufsMmax.textContent = fmtLU(r.lufsM[r.mMaxIdx]);
    els.tMmax.textContent = fmtTime(r.timesM[r.mMaxIdx]);
  } else {
    els.lufsMmax.textContent = '—';
    els.tMmax.textContent = '—';
  }
  if (r.sMaxIdx >= 0) {
    els.lufsSmax.textContent = fmtLU(r.lufsS[r.sMaxIdx]);
    els.tSmax.textContent = fmtTime(r.timesS[r.sMaxIdx]);
  } else {
    els.lufsSmax.textContent = '—';
    els.tSmax.textContent = '—';
  }

  els.lra.textContent = fmtLU(r.lra);
  els.dbtp.textContent = fmtDb(r.dbtp);
  els.plr.textContent = fmtLU(r.plr);
  els.dr.textContent = fmtLU(r.dr);

  // Charts
  drawSeries(els.canvasM, r.timesM, r.lufsM);
  drawSeries(els.canvasS, r.timesS, r.lufsS);
}

// ---------- Live (mic) with AudioWorklet ----------

async function toggleLive() {
  if (live.running) {
    // stop
    live.workletNode?.port?.postMessage({ type: 'stop' });
    live.workletNode?.disconnect();
    live.hs?.disconnect();
    live.hp?.disconnect();
    if (live.stream) {
      live.stream.getTracks().forEach(t => t.stop());
    }
    live.ctx?.close();
    live = { ctx: null, stream: null, workletNode: null, hp: null, hs: null, running: false };
    els.liveBtn.textContent = 'Live (mic)';
    logln('Live detenido.');
    return;
  }

  // start
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: { ideal: 2 }, noiseSuppression: false, echoCancellation: false, autoGainControl: false } });
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    await ctx.audioWorklet.addModule('worklet-processor.js');
    const src = ctx.createMediaStreamSource(stream);

    // K-weighting nodes (main thread to keep worklet simple)
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = 60;
    hp.Q.value = Math.SQRT1_2;

    const hs = ctx.createBiquadFilter();
    hs.type = 'highshelf';
    hs.frequency.value = 4000;
    hs.gain.value = 4;
    hs.Q.value = 0.707;

    const worklet = new AudioWorkletNode(ctx, 'loudness-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      outputChannelCount: [],
      processorOptions: {
        sampleRate: ctx.sampleRate,
        mWindowSec: M_WINDOW_SEC,
        sWindowSec: S_WINDOW_SEC,
        hopSec: DEFAULT_HOP_SEC,
        kOffsetDb: K_OFFSET_DB,
      }
    });

    worklet.port.onmessage = (ev) => {
      const d = ev.data || {};
      if (d.type === 'metrics') {
        // live instantaneous updates
        els.lufsM.textContent = fmtLU(d.M.current);
        els.lufsS.textContent = fmtLU(d.S.current);
        els.lufsI.textContent = fmtLU(d.integrated);
        els.dbtp.textContent = fmtDb(d.dbtp);
        els.plr.textContent = fmtLU(d.dbtp - d.integrated);
        els.lra.textContent = fmtLU(d.lra);
        els.dr.textContent = fmtLU(d.dr);

        // Draw partial series
        if (d.M.times && d.M.values) drawSeries(els.canvasM, d.M.times, d.M.values);
        if (d.S.times && d.S.values) drawSeries(els.canvasS, d.S.times, d.S.values);

        // Max times
        if (typeof d.M.maxIdx === 'number' && d.M.times?.length) {
          els.lufsMmax.textContent = fmtLU(d.M.values[d.M.maxIdx]);
          els.tMmax.textContent = fmtTime(d.M.times[d.M.maxIdx]);
        }
        if (typeof d.S.maxIdx === 'number' && d.S.times?.length) {
          els.lufsSmax.textContent = fmtLU(d.S.values[d.S.maxIdx]);
          els.tSmax.textContent = fmtTime(d.S.times[d.S.maxIdx]);
        }
      } else if (d.type === 'log') {
        logln('[live]', d.msg);
      }
    };

    src.connect(hp).connect(hs).connect(worklet);

    live = { ctx, stream, workletNode: worklet, hp, hs, running: true };
    els.liveBtn.textContent = 'Detener Live';
    logln('Live iniciado @', ctx.sampleRate, 'Hz');
  } catch (err) {
    console.error(err);
    alert('No se pudo iniciar el modo Live: ' + err.message);
  }
}

// ---------- Event bindings ----------

els.analyzeBtn.addEventListener('click', async () => {
  try {
    const file = els.fileInput.files?.[0];
    await analyzeFile(file);
  } catch (err) {
    console.error(err);
    logln('Error:', err.message);
    alert('Error al analizar: ' + err.message);
  }
});

els.liveBtn.addEventListener('click', toggleLive);

// Light/Dark manual toggle (forces color-scheme via data-theme attr)
els.darkToggle.addEventListener('change', (e) => {
  if (e.target.checked) {
    document.documentElement.style.colorScheme = 'dark';
  } else {
    document.documentElement.style.colorScheme = 'light';
  }
});

// ---------- Test helpers (callable from console) ----------

async function computeFromChannels(channels, sampleRate) {
  const w = ensureWorker();
  return await new Promise((resolve, reject) => {
    const onMsg = (ev) => {
      if (ev.data?.ok) { w.removeEventListener('message', onMsg); resolve(ev.data.result); }
      else if (ev.data?.ok === false) { w.removeEventListener('message', onMsg); reject(new Error(ev.data?.error || 'Worker error')); }
    };
    w.addEventListener('message', onMsg);
    w.postMessage({
      type: 'compute',
      payload: {
        channels, sampleRate,
        hopSec: DEFAULT_HOP_SEC,
        mWindowSec: M_WINDOW_SEC,
        sWindowSec: S_WINDOW_SEC,
      }
    });
  });
}

function genSilence(durationSec = 5, sampleRate = 48000, nCh = 2) {
  const N = Math.floor(durationSec * sampleRate);
  const ch = [];
  for (let c = 0; c < nCh; c++) ch.push(new Float32Array(N));
  return { ch, sampleRate };
}
function genSine(freq = 1000, amp = 0.5, durationSec = 3, sampleRate = 48000, nCh = 2) {
  const N = Math.floor(durationSec * sampleRate);
  const ch = [];
  for (let c = 0; c < nCh; c++) {
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = amp * Math.sin(2*Math.PI*freq*n/sampleRate);
    ch.push(x);
  }
  return { ch, sampleRate };
}
function genNoise(amp = 0.25, durationSec = 5, sampleRate = 48000, nCh = 2) {
  const N = Math.floor(durationSec * sampleRate);
  const ch = [];
  for (let c = 0; c < nCh; c++) {
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = amp * (Math.random()*2 - 1);
    ch.push(x);
  }
  return { ch, sampleRate };
}
function genFadeMusicLike(durationSec = 6, sampleRate = 48000, nCh = 2) {
  // simple synth: sum of sines with fade in/out
  const N = Math.floor(durationSec * sampleRate);
  const ch = [];
  for (let c = 0; c < nCh; c++) {
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      const t = n / sampleRate;
      const env = Math.min(1, Math.max(0, Math.min(t / 1.5, (durationSec - t) / 1.5)));
      const sig = 0.4*Math.sin(2*Math.PI*220*t) + 0.25*Math.sin(2*Math.PI*440*t) + 0.15*Math.sin(2*Math.PI*880*t);
      x[n] = env * sig;
    }
    ch.push(x);
  }
  return { ch, sampleRate };
}

function ok(cond, msg) { const s = cond ? 'OK' : 'FAIL'; logln(s + ' - ' + msg); return cond; }

window.tests = {
  async silence() {
    const { ch, sampleRate } = genSilence(5);
    const r = await computeFromChannels(ch, sampleRate);
    const conds = [
      r.lufsI < -60,
      r.dbtp < -60,
      r.lra <= 1,
      Array.isArray(r.lufsM) && Array.isArray(r.lufsS)
    ];
    ok(conds[0], 'Silencio: LUFS-I < -60 ('+fmtLU(r.lufsI)+')');
    ok(conds[1], 'Silencio: dBTP < -60 ('+fmtDb(r.dbtp)+')');
    ok(conds[2], 'Silencio: LRA ≤ 1 ('+fmtLU(r.lra)+')');
    ok(conds[3], 'Silencio: series M/S creadas');
    return r;
  },
  async sine() {
    const { ch, sampleRate } = genSine(1000, 0.5, 3);
    const r = await computeFromChannels(ch, sampleRate);
    ok(r.lufsM.length > 0 && r.lufsS.length > 0, 'Seno: series no vacías');
    ok((r.dbtp - r.lufsI) > 0, 'Seno: PLR > 0 (PLR=' + fmtLU(r.dbtp - r.lufsI)+')');
    ok(r.mMaxIdx >= 0 && r.sMaxIdx >= 0, 'Seno: índices de máximos válidos');
    ok(r.timesM[r.mMaxIdx] >= 0 && r.timesM[r.mMaxIdx] <= 3.0, 'Seno: tiempo Mmax dentro de duración (' + fmtTime(r.timesM[r.mMaxIdx]) + ')');
    ok(r.timesS[r.sMaxIdx] >= 0 && r.timesS[r.sMaxIdx] <= 3.0, 'Seno: tiempo Smax dentro de duración (' + fmtTime(r.timesS[r.sMaxIdx]) + ')');
    return r;
  },
  async noise() {
    const { ch, sampleRate } = genNoise(0.25, 5);
    const r = await computeFromChannels(ch, sampleRate);
    ok(r.lra > 0, 'Ruido: LRA > 0 (' + fmtLU(r.lra) + ')');
    ok(isFinite(r.lufsI) && isFinite(r.dbtp), 'Ruido: valores finitos');
    return r;
  },
  async fade() {
    const { ch, sampleRate } = genFadeMusicLike(8);
    const r = await computeFromChannels(ch, sampleRate);
    ok(r.lra > 0, 'Fade: LRA > 0 (' + fmtLU(r.lra) + ')');
    ok(r.dr > 0, 'Fade: DR > 0 (' + fmtLU(r.dr) + ')');
    return r;
  }
};

// Expose for convenience in console
window._dbg = { analyzeFile, renderKWeighted, computeFromChannels };

// Optional: analyze immediately if a file was selected
els.fileInput.addEventListener('change', () => { /* no-op */ });
