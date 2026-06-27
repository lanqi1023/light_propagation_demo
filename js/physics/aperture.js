// ─── Aperture Generation ───
// All functions write interleaved complex values into out array (Float64Array, len = nx*ny*2)
// out[i*2] = Re, out[i*2+1] = Im

const Aperture = {};

// Coordinate grid helpers (called once per generate)
function makeGrids(nx, ny, dx, dy) {
  const xc = (nx - 1) / 2;
  const yc = (ny - 1) / 2;
  const xs = new Float64Array(nx);
  const ys = new Float64Array(ny);
  for (let i = 0; i < nx; i++) xs[i] = (i - xc) * dx;
  for (let j = 0; j < ny; j++) ys[j] = (j - yc) * dy;
  return { xs, ys };
}

Aperture.generate = function(type, out, nx, ny, dx, dy, params, w0) {
  // Reset to zeros
  out.fill(0);

  const { xs, ys } = makeGrids(nx, ny, dx, dy);

  switch (type) {
    case 'slit':
      slit(out, nx, ny, xs, ys, params.width || 100);
      break;
    case 'doubleSlit':
      doubleSlit(out, nx, ny, xs, params.width || 50, params.separation || 200);
      break;
    case 'circle':
      circle(out, nx, ny, xs, ys, params.radius || 100);
      break;
    case 'rectangle':
      rectangle(out, nx, ny, xs, ys, params.width || 150, params.height || 150);
      break;
    case 'free':
      freeSpace(out, nx, ny);
      break;
    case 'upload':
      // Image data loaded separately; just use current out as-is (preset by main thread)
      break;
    default:
      // default to slit
      slit(out, nx, ny, xs, ys, params.width || 100);
  }

  // Apply Gaussian envelope (w0 ≥ 100mm → effectively plane wave, skip)
  if (w0 && w0 > 0 && w0 < 100) {
    const w0sq = w0 * w0 * 1e6; // mm^2 → μm^2
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const r2 = xs[i] * xs[i] + ys[j] * ys[j];
        const g = Math.exp(-r2 / w0sq);
        const idx = (i * ny + j) * 2;
        out[idx]     *= g;
        out[idx + 1] *= g;
      }
    }
  }
};

function slit(out, nx, ny, xs, ys, width) {
  const halfW = width / 2;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if (Math.abs(xs[i]) <= halfW) {
        const idx = (i * ny + j) * 2;
        out[idx] = 1.0;  // Re = 1
      }
    }
  }
}

function doubleSlit(out, nx, ny, xs, width, separation) {
  const halfW = width / 2;
  const halfS = separation / 2;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const x = xs[i];
      if ((Math.abs(x - halfS) <= halfW) || (Math.abs(x + halfS) <= halfW)) {
        const idx = (i * ny + j) * 2;
        out[idx] = 1.0;
      }
    }
  }
}

function circle(out, nx, ny, xs, ys, radius) {
  const r2 = radius * radius;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if (xs[i] * xs[i] + ys[j] * ys[j] <= r2) {
        const idx = (i * ny + j) * 2;
        out[idx] = 1.0;
      }
    }
  }
}

function rectangle(out, nx, ny, xs, ys, width, height) {
  const halfW = width / 2;
  const halfH = height / 2;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if (Math.abs(xs[i]) <= halfW && Math.abs(ys[j]) <= halfH) {
        const idx = (i * ny + j) * 2;
        out[idx] = 1.0;
      }
    }
  }
}

function freeSpace(out, nx, ny) {
  for (let i = 0; i < nx * ny * 2; i += 2) {
    out[i] = 1.0;
  }
}
