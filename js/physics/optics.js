// ─── Non-ideal Factors (Phase 4) ───
// Stub: real implementations added in M11-M13

const Optics = {};

// Apply tilt phase: U *= exp(i * k * (sinθx·x + sinθy·y))
Optics.applyTilt = function(U, nx, ny, dx, dy, lambda, thetaXDeg, thetaYDeg) {
  if (!thetaXDeg && !thetaYDeg) return;
  const lambdaUm = lambda / 1000;
  const k = 2 * Math.PI / lambdaUm;
  const thetaXRad = thetaXDeg * Math.PI / 180;
  const thetaYRad = thetaYDeg * Math.PI / 180;
  const sinTX = Math.sin(thetaXRad);
  const sinTY = Math.sin(thetaYRad);

  const xc = (nx - 1) / 2;
  const yc = (ny - 1) / 2;

  for (let i = 0; i < nx; i++) {
    const x = (i - xc) * dx;
    for (let j = 0; j < ny; j++) {
      const y = (j - yc) * dy;
      const phase = k * (sinTX * x + sinTY * y);
      const idx = (i * ny + j) * 2;
      const re = U[idx], im = U[idx + 1];
      const c = Math.cos(phase), s = Math.sin(phase);
      U[idx]     = re * c - im * s;
      U[idx + 1] = re * s + im * c;
    }
  }
};

// Multi-wavelength incoherent sum (temporal coherence)
// Returns intensity array (Uint8Array, normalized 0-255)
Optics.temporalCoherence = function(U0_template, nx, ny, dx, dy, lambda0, deltaLambda, M, dz, work, usePadding, bandLimited) {
  // Stub: will sum over M wavelengths
  // For now return single-wavelength result
  return null;
};

// Multi-angle incoherent sum (spatial coherence, 1D X direction)
Optics.spatialCoherence = function(/*...*/) {
  return null;
};
