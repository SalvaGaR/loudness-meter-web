/* global registerProcessor */

/**
 * AudioWorkletProcessor that:
 * - recibe audio K-weighted (desde main: highpass -> highshelf -> worklet)
 * - acumula y emite cada hop (100 ms) valores Momentary y Short-term
 * - calcula "en vivo" Integrated (con gating simplificado), LRA y DR sobre lo acumulado
 * - estima True Peak (aprox) con oversampling 4× sobre el último bloque
 *
 * Nota: El cálculo live es "running" (sobre lo recibido hasta ahora). El análisis por archivo
 * es más preciso porque procesa todo el buffer offline.
 */

class LoudnessProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options?.processorOptions || {};
    this.sampleRate = o.sampleRate || sampleRate;
    this.mWin = Math.max(1, Math.floor((o.mWindowSec || 0.4) * this.sampleRate));
    this.sWin = Math.max(1, Math.floor((o.sWindowSec || 3.0) * this.sampleRate));
    this.hop = Math.max(1, Math.floor((o.hopSec || 0.1) * this.sampleRate));
    this.K_OFFSET_DB = o.kOffsetDb ?? -0.691;

    this.frameCount = 0;
    this.buf = [];      // multi-channel circular buffer for last S window
    this.bufM = [];     // for last M window
    this.bufSize = this.sWin;
    this.bufMSize = this.mWin;

    this.acc = {
      // series over time (for drawing)
      timesM: [],
      lufsM: [],
      timesS: [],
      lufsS: [],
      msM: [], // for gating integrated
    };

    this.timeSec = 0;
    this.lastHopFrame = 0;

    this.port.onmessage = (ev) => {
      const d = ev.data || {};
      if (d.type === 'stop') {
        // no-op by now
      }
    };
  }

  // Helpers
  lufsFromMS(ms) {
    if (ms <= 0) return Number.NEGATIVE_INFINITY;
    return 10 * Math.log10(ms) + this.K_OFFSET_DB;
  }
  percentile(sorted, p) {
    if (!sorted.length) return NaN;
    if (p <= 0) return sorted[0];
    if (p >= 100) return sorted[sorted.length - 1];
    const pos = (p/100) * (sorted.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    const frac = pos - lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
  }
  truePeak4xBlock(block) {
    // block: Float32Array mono (mix of channels)
    let peak = 0;
    const x = block;
    const N = x.length;
    for (let i = 0; i < N; i++) {
      const a = Math.abs(x[i]); if (a > peak) peak = a;
    }
    for (let i = 0; i < N - 1; i++) {
      const s0 = i > 0 ? x[i-1] : x[i];
      const s1 = x[i];
      const s2 = x[i+1];
      const s3 = (i+2 < N) ? x[i+2] : x[i+1];
      for (let k = 1; k < 4; k++) {
        const t = k * 0.25;
        const a = 2*s1;
        const b = -s0 + s2;
        const c = 2*s0 - 5*s1 + 4*s2 - s3;
        const d = -s0 + 3*s1 - 3*s2 + s3;
        const y = 0.5 * (a + b*t + c*t*t + d*t*t*t);
        const ab = Math.abs(y);
        if (ab > peak) peak = ab;
      }
    }
    return (peak > 0) ? 20*Math.log10(peak) : Number.NEGATIVE_INFINITY;
  }

  process(inputs /* [[Float32Array[ch]]] */) {
    const input = inputs[0];
    if (!input || !input.length || !input[0].length) return true;

    const nCh = input.length;
    const N = input[0].length;

    // ensure buffers
    if (this.buf.length !== nCh) {
      this.buf = Array.from({ length: nCh }, () => new Float32Array(this.bufSize));
      this.bufM = Array.from({ length: nCh }, () => new Float32Array(this.bufMSize));
      this.writeIdx = 0;
      this.writeIdxM = 0;
      this.filled = 0;
      this.filledM = 0;
    }

    // write incoming to circular buffers
    for (let c = 0; c < nCh; c++) {
      const x = input[c];
      // S buffer
      for (let i = 0; i < N; i++) {
        this.buf[c][(this.writeIdx + i) % this.bufSize] = x[i];
      }
      // M buffer
      for (let i = 0; i < N; i++) {
        this.bufM[c][(this.writeIdxM + i) % this.bufMSize] = x[i];
      }
    }
    this.writeIdx = (this.writeIdx + N) % this.bufSize;
    this.writeIdxM = (this.writeIdxM + N) % this.bufMSize;
    this.filled = Math.min(this.filled + N, this.bufSize);
    this.filledM = Math.min(this.filledM + N, this.bufMSize);

    // hop trigger each 'hop' frames
    this.frameCount += N;
    this.timeSec += N / this.sampleRate;

    if ((this.frameCount - this.lastHopFrame) >= this.hop) {
      this.lastHopFrame = this.frameCount;

      // compute MS over M and S windows
      const msM = this.meanSquareOverBuffer(this.bufM, this.writeIdxM, this.filledM, this.bufMSize);
      const msS = this.meanSquareOverBuffer(this.buf, this.writeIdx, this.filled, this.bufSize);

      const lufsM = this.lufsFromMS(msM);
      const lufsS = this.lufsFromMS(msS);

      const t = this.timeSec;

      this.acc.timesM.push(t);
      this.acc.lufsM.push(lufsM);
      this.acc.msM.push(msM);

      this.acc.timesS.push(t);
      this.acc.lufsS.push(lufsS);

      // Integrated with gating over accumulated M blocks
      const ABS_GATE = -70;
      const keptAbs = [];
      const keptAbsMS = [];
      for (let i = 0; i < this.acc.lufsM.length; i++) {
        if (this.acc.lufsM[i] > ABS_GATE) { keptAbs.push(this.acc.lufsM[i]); keptAbsMS.push(this.acc.msM[i]); }
      }
      let integrated = Number.NEGATIVE_INFINITY;
      let relGateThr = Number.NEGATIVE_INFINITY;
      if (keptAbsMS.length) {
        const meanAbs = keptAbsMS.reduce((a,b)=>a+b,0) / keptAbsMS.length;
        const prelim = 10*Math.log10(meanAbs) + this.K_OFFSET_DB;
        relGateThr = prelim - 10;
        const keptRel = [];
        for (let i = 0; i < this.acc.lufsM.length; i++) {
          if (this.acc.lufsM[i] >= relGateThr) keptRel.push(this.acc.msM[i]);
        }
        if (keptRel.length) {
          const meanRel = keptRel.reduce((a,b)=>a+b,0) / keptRel.length;
          integrated = 10*Math.log10(meanRel) + this.K_OFFSET_DB;
        } else {
          integrated = prelim;
        }
      }

      // LRA P10..P95 on S with rel gate −20 LU from integrated
      let lra = 0, dr = 0, dbtp = -Infinity;
      if (this.acc.lufsS.length) {
        const thr = integrated - 20;
        const base = this.acc.lufsS.filter(v => isFinite(v) && v >= thr);
        const arr = (base.length ? base : this.acc.lufsS.filter(isFinite)).slice().sort((a,b)=>a-b);
        const p10 = this.percentile(arr, 10);
        const p95 = this.percentile(arr, 95);
        lra = p95 - p10;

        const arrAll = this.acc.lufsS.filter(isFinite).slice().sort((a,b)=>a-b);
        const p5 = this.percentile(arrAll, 5);
        const p95b = this.percentile(arrAll, 95);
        dr = p95b - p5;
      }

      // Approx true-peak on the latest hop chunk (mixdown mono)
      // Mix to mono
      const hopN = Math.min(this.hop, this.filledM); // approximate last hop size
      const mono = new Float32Array(hopN);
      for (let i = 0; i < hopN; i++) {
        let s = 0;
        for (let c = 0; c < nCh; c++) s += this.bufM[c][(this.writeIdxM - hopN + i + this.bufMSize) % this.bufMSize];
        mono[i] = s; // sum, consistent with channel energy sum
      }
      dbtp = this.truePeak4xBlock(mono);

      // Max indices for UI markers
      const mMaxIdx = this.idxMax(this.acc.lufsM);
      const sMaxIdx = this.idxMax(this.acc.lufsS);

      this.port.postMessage({
        type: 'metrics',
        M: { current: lufsM, times: this.acc.timesM, values: this.acc.lufsM, maxIdx: mMaxIdx },
        S: { current: lufsS, times: this.acc.timesS, values: this.acc.lufsS, maxIdx: sMaxIdx },
        integrated,
        lra, dr, dbtp
      });
    }

    return true;
  }

  meanSquareOverBuffer(bufs, writeIdx, filled, size) {
    if (filled <= 0) return 0;
    // average per-sample squared per channel, then sum channel MS (BS.1770)
    let sumMS = 0;
    for (let c = 0; c < bufs.length; c++) {
      const b = bufs[c];
      let acc = 0;
      // use last 'filled' samples
      for (let i = 0; i < filled; i++) {
        const idx = (writeIdx - filled + i + size) % size;
        const v = b[idx];
        acc += v*v;
      }
      sumMS += acc / filled;
    }
    return sumMS;
  }

  idxMax(arr) {
    let m = -Infinity, idx = -1;
    for (let i = 0; i < arr.length; i++) { if (arr[i] > m) { m = arr[i]; idx = i; } }
    return idx;
  }
}

registerProcessor('loudness-processor', LoudnessProcessor);
