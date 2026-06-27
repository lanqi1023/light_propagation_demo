// ─── 2D FFT Module ───
// Data layout: Float64Array, interleaved [Re0, Im0, Re1, Im1, ...]
// Row-major: index = (i * ncols + j) * 2

const FFT = {};

// ─── Pre-computed tables (rebuilt when size changes) ───
let _fftCache = { n: 0 };

function ensureFFTTables(n) {
  if (_fftCache.n === n) return;
  const bitrev = new Uint32Array(n);
  const twiddle = new Float64Array(n * 2);  // re, im pairs

  // bit-reversal permutation
  let bits = 0;
  while ((1 << bits) < n) bits++;
  for (let i = 0; i < n; i++) {
    let rev = 0;
    for (let b = 0; b < bits; b++) {
      if (i & (1 << b)) rev |= (1 << (bits - 1 - b));
    }
    bitrev[i] = rev;
  }

  // twiddle factors W_n^k = exp(-i*2π*k/n)
  for (let k = 0; k < n / 2; k++) {
    const angle = -2 * Math.PI * k / n;
    twiddle[k * 2]     = Math.cos(angle);
    twiddle[k * 2 + 1] = Math.sin(angle);
  }

  _fftCache = { n, bitrev, twiddle, bits };
}

// ─── 1D FFT (in-place, interleaved complex) ───
// arr: Float64Array, length = n * 2
// n: must be power of 2
// inverse: boolean
function fft1d(arr, n, inverse) {
  ensureFFTTables(n);
  const { bitrev, twiddle } = _fftCache;

  // bit-reversal permutation
  for (let i = 0; i < n; i++) {
    const j = bitrev[i];
    if (i < j) {
      const i2 = i * 2, j2 = j * 2;
      const tr = arr[i2], ti = arr[i2 + 1];
      arr[i2]     = arr[j2];
      arr[i2 + 1] = arr[j2 + 1];
      arr[j2]     = tr;
      arr[j2 + 1] = ti;
    }
  }

  // Cooley-Tukey butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < half; j++) {
        const tIdx = j * step;
        const wr = twiddle[tIdx * 2];
        const wi = inverse ? -twiddle[tIdx * 2 + 1] : twiddle[tIdx * 2 + 1];

        const evenIdx = (i + j) * 2;
        const oddIdx  = (i + j + half) * 2;

        const tr = arr[oddIdx] * wr - arr[oddIdx + 1] * wi;
        const ti = arr[oddIdx] * wi + arr[oddIdx + 1] * wr;

        arr[oddIdx]     = arr[evenIdx]     - tr;
        arr[oddIdx + 1] = arr[evenIdx + 1] - ti;
        arr[evenIdx]     = arr[evenIdx]     + tr;
        arr[evenIdx + 1] = arr[evenIdx + 1] + ti;
      }
    }
  }

  // inverse: scale by 1/n
  if (inverse) {
    const invN = 1 / n;
    for (let i = 0; i < n * 2; i++) {
      arr[i] *= invN;
    }
  }
}

// ─── 2D FFT ───
// arr: Float64Array, length = nx * ny * 2
// Row-major: element (i,j) at index (i * ny + j) * 2
function fft2d(arr, nx, ny, inverse) {
  // FFT each row
  for (let i = 0; i < nx; i++) {
    const row = arr.subarray(i * ny * 2, (i + 1) * ny * 2);
    fft1d(row, ny, inverse);
  }
  // transpose + FFT each column (now rows)
  const tmp = new Float64Array(Math.max(nx, ny) * 2);
  for (let j = 0; j < ny; j++) {
    // copy column j to tmp
    for (let i = 0; i < nx; i++) {
      tmp[i * 2]     = arr[(i * ny + j) * 2];
      tmp[i * 2 + 1] = arr[(i * ny + j) * 2 + 1];
    }
    fft1d(tmp, nx, inverse);
    // write back
    for (let i = 0; i < nx; i++) {
      arr[(i * ny + j) * 2]     = tmp[i * 2];
      arr[(i * ny + j) * 2 + 1] = tmp[i * 2 + 1];
    }
  }
}

// ─── fftshift: zero-frequency to center ───
function fftshift(arr, nx, ny) {
  const halfX = nx >> 1;
  const halfY = ny >> 1;
  const tmp = new Float64Array(2);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const iSwap = (i + halfX) % nx;
      const jSwap = (j + halfY) % ny;
      if (i < iSwap || (i === iSwap && j < jSwap)) {
        const idx = (i * ny + j) * 2;
        const sIdx = (iSwap * ny + jSwap) * 2;
        tmp[0] = arr[idx]; tmp[1] = arr[idx + 1];
        arr[idx] = arr[sIdx]; arr[idx + 1] = arr[sIdx + 1];
        arr[sIdx] = tmp[0]; arr[sIdx + 1] = tmp[1];
      }
    }
  }
}

// ─── ifftshift: inverse of fftshift ───
function ifftshift(arr, nx, ny) {
  // same as fftshift for even dimensions
  fftshift(arr, nx, ny);
}

// ─── Self-test (runs on Worker init) ───
function fftSelfTest() {
  const n = 8;
  const arr = new Float64Array(n * 2);
  // impulse: [1,0, 0,0, ...]
  arr[0] = 1; arr[1] = 0;
  fft1d(arr, n, false);
  let ok = true;
  for (let i = 0; i < n * 2; i++) {
    if (Math.abs(arr[i] - (i % 2 === 0 ? 1 : 0)) > 1e-10) ok = false;
  }
  if (!ok) { self.postMessage({ type: 'fftSelfTest', passed: false, msg: 'impulse test failed' }); return; }

  // random array roundtrip
  const n2 = 16;
  const a = new Float64Array(n2 * 2);
  const orig = new Float64Array(n2 * 2);
  for (let i = 0; i < n2 * 2; i++) {
    a[i] = Math.random() * 2 - 1;
    orig[i] = a[i];
  }
  fft1d(a, n2, false);
  fft1d(a, n2, true);
  let maxErr = 0;
  for (let i = 0; i < n2 * 2; i++) {
    const err = Math.abs(a[i] - orig[i]);
    if (err > maxErr) maxErr = err;
  }
  if (maxErr > 1e-12) {
    self.postMessage({ type: 'fftSelfTest', passed: false, msg: 'roundtrip error: ' + maxErr });
    return;
  }

  self.postMessage({ type: 'fftSelfTest', passed: true, msg: 'FFT self-test OK, maxErr=' + maxErr.toExponential(2) });
}

// Run self-test
fftSelfTest();
