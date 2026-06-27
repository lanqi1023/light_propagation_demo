// Load sub-modules
importScripts('math/fft.js', 'physics/aperture.js', 'physics/angspec.js', 'physics/optics.js');

// ─── Memory Pool ───
let pool = {};

function ensurePool(nx, ny) {
  const needsRealloc = !pool.U0 || pool.Nx !== nx || pool.Ny !== ny;
  if (!needsRealloc) return;
  const n2 = nx * ny * 2;
  pool = {
    Nx: nx, Ny: ny,
    U0:   new Float64Array(n2),   // input field
    Uz:   new Float64Array(n2),   // output field
    work: new Float64Array(n2),   // scratch
  };
}

// ─── Uploaded image cache ───
let _uploadedPixels = null;
let _uploadedNx = 0, _uploadedNy = 0;

// ─── Message Handler ───
self.onmessage = function(e) {
  const msg = e.data;
  switch (msg.type) {

    case 'ping':
      self.postMessage({ type: 'pong' });
      break;

    case 'uploadImage':
      _uploadedPixels = msg.pixels;
      _uploadedNx = msg.nx;
      _uploadedNy = msg.ny;
      self.postMessage({ type: 'uploadAccepted', nx: _uploadedNx, ny: _uploadedNy });
      break;

    case 'compute':
      handleCompute(msg);
      break;

    default:
      self.postMessage({ type: 'error', message: 'unknown message type: ' + msg.type });
  }
};

function handleCompute(msg) {
  const t0 = performance.now();
  const p = msg.params;
  const nx = p.Nx, ny = p.Ny, dx = p.dx, dy = p.dy;
  ensurePool(nx, ny);

  // 1. Generate aperture (or load uploaded image)
  if (p.apertureType === 'upload' && _uploadedPixels) {
    for (let k = 0; k < nx * ny; k++) {
      const amp = k < _uploadedPixels.length ? _uploadedPixels[k] : 0;
      pool.U0[k * 2] = amp;
      pool.U0[k * 2 + 1] = 0;
    }
  } else {
    Aperture.generate(p.apertureType, pool.U0, nx, ny, dx, dy, p.apertureParams, p.w0);
  }

  // 1b. Apply tilt if enabled
  if (p.tiltOn) {
    Optics.applyTilt(pool.U0, nx, ny, dx, dy, p.lambda, p.tiltX, p.tiltY);
  }

  // 2. Propagate
  AngSpec.propagate(pool.U0, pool.Uz, nx, ny, dx, dy, p.lambda, p.dz, pool.work);

  // 3. Extract intensity + phase
  const intensity_arr = new Uint8Array(nx * ny);
  const phase_arr     = new Uint8Array(nx * ny);

  let maxInt = 0;
  for (let i = 0; i < nx * ny; i++) {
    const re = pool.Uz[i * 2];
    const im = pool.Uz[i * 2 + 1];
    const I = re * re + im * im;
    intensity_arr[i] = 0; // filled after normalization pass
    phase_arr[i] = 0;
    if (I > maxInt) maxInt = I;
  }
  // scale, clamp
  for (let i = 0; i < nx * ny; i++) {
    const re = pool.Uz[i * 2];
    const im = pool.Uz[i * 2 + 1];
    const I = re * re + im * im;
    const scaled = maxInt > 0 ? Math.round(255 * Math.sqrt(I / maxInt)) : 0;
    intensity_arr[i] = Math.min(255, Math.max(0, scaled));
    phase_arr[i] = Math.round(((Math.atan2(im, re) / Math.PI + 1) / 2) * 255) & 0xFF;
  }

  // 4. Cross-sections (center row & column)
  const cRow = new Float64Array(ny);
  const cCol = new Float64Array(nx);
  const cy = nx >> 1;
  for (let j = 0; j < ny; j++) {
    const re = pool.Uz[cy * ny * 2 + j * 2];
    const im = pool.Uz[cy * ny * 2 + j * 2 + 1];
    cRow[j] = re * re + im * im;
  }
  const cx = ny >> 1;
  for (let i = 0; i < nx; i++) {
    const re = pool.Uz[i * ny * 2 + cx * 2];
    const im = pool.Uz[i * ny * 2 + cx * 2 + 1];
    cCol[i] = re * re + im * im;
  }

  // 5. Sampling info
  const Lx = nx * dx / 1000;   // mm
  const Ly = ny * dy / 1000;
  const fresnelNum = Lx * Lx / (4 * p.lambda * 1e-6 * p.dz);
  const samplingOk = dx <= p.lambda / 2000; // λ/2 in mm → λ/2000 in μm? no.
  // Actually λ is in nm, dx is in μm. λ/2 in μm = λ / 2000.
  // So dx <= λ/2000 means dx in μm <= λ_nm / 2000
  // λ=532nm → λ/2 = 266nm = 0.266μm
  const dxUm = dx, lambdaNm = p.lambda;
  const nyquistOk = dxUm <= lambdaNm / 2000;

  const t1 = performance.now();

  self.postMessage({
    type: 'result',
    id: msg.id,
    intensity: intensity_arr,
    phase: phase_arr,
    crossX: cRow,
    crossY: cCol,
    info: {
      fresnelNum: fresnelNum,
      Lx: Lx, Ly: Ly,
      nx: nx, ny: ny,
      computationTime: Math.round(t1 - t0),
      samplingOk: nyquistOk,
      maxInt: maxInt,
    }
  }, [
    intensity_arr.buffer,
    phase_arr.buffer,
    cRow.buffer,
    cCol.buffer,
  ]);
}
