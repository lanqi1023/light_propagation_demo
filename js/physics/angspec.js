// ─── Angular Spectrum Method ───
// Band-limited ASM (Matsushima 2009) + Zero-padding

const AngSpec = {};

// Temporary work buffer for padded grid (reused)
let _padBuf = null;
let _padNx = 0, _padNy = 0;

function ensurePad(nx, ny) {
  const pnx = nx * 2, pny = ny * 2;
  if (!_padBuf || _padBuf.length < pnx * pny * 2 || _padNx !== nx || _padNy !== ny) {
    _padBuf = new Float64Array(pnx * pny * 2);
    _padNx = nx; _padNy = ny;
  }
}

/**
 * Propagate input field using angular spectrum method
 * @param {Float64Array} U0   - input field (nx × ny, interleaved complex)
 * @param {Float64Array} Uz   - output field (nx × ny, interleaved complex, pre-allocated)
 * @param {number} nx, ny     - grid dimensions
 * @param {number} dx, dy     - pixel pitch (μm)
 * @param {number} lambda     - wavelength (nm)
 * @param {number} dz         - propagation distance (mm)
 * @param {Float64Array} work - scratch buffer (same size as U0)
 * @param {boolean} usePadding - enable 2× zero-padding (default true)
 * @param {boolean} bandLimited - enable Matsushima band-limiting (default true)
 */
AngSpec.propagate = function(U0, Uz, nx, ny, dx, dy, lambda, dz, work, usePadding, bandLimited) {
  usePadding = usePadding !== false;
  bandLimited = bandLimited !== false;

  const lambdaUm = lambda / 1000; // nm → μm
  const dzUm = dz * 1000;         // mm → μm

  if (usePadding) {
    // ─── 2× Zero-padded path ───
    const pnx = nx * 2, pny = ny * 2;
    const Lx = nx * dx, Ly = ny * dy;
    ensurePad(nx, ny);

    // Clear padded buffer
    _padBuf.fill(0);

    // Place U0 at center of padded grid
    const ox = (pnx - nx) >> 1;  // offset in rows
    const oy = (pny - ny) >> 1;  // offset in cols
    for (let i = 0; i < nx; i++) {
      const srcRow = i * ny * 2;
      const dstRow = (ox + i) * pny * 2 + oy * 2;
      for (let j = 0; j < ny; j++) {
        _padBuf[dstRow + j * 2]     = U0[srcRow + j * 2];
        _padBuf[dstRow + j * 2 + 1] = U0[srcRow + j * 2 + 1];
      }
    }

    // FFT on padded grid
    ifftshift(_padBuf, pnx, pny);
    fft2d(_padBuf, pnx, pny, false);
    fftshift(_padBuf, pnx, pny);

    // Build and apply transfer function
    const k = 2 * Math.PI / lambdaUm;
    for (let m = 0; m < pnx; m++) {
      // fx: fftshift convention, zero at center
      const fm = m < pnx / 2 ? m : m - pnx;
      const fx = fm / (pnx * dx);
      for (let n = 0; n < pny; n++) {
        const fn = n < pny / 2 ? n : n - pny;
        const fy = fn / (pny * dy);

        const kx = 2 * Math.PI * fx;
        const ky = 2 * Math.PI * fy;
        const kxy2 = kx * kx + ky * ky;

        let H_re = 0, H_im = 0;

        if (kxy2 <= k * k) {
          // Propagating modes
          const kz = Math.sqrt(k * k - kxy2);

          // Band-limiting (Matsushima 2009)
          if (bandLimited) {
            const fxMax = 1 / (2 * dx);
            const fyMax = 1 / (2 * dy);
            const fxLimit = fxMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / (Lx * Lx), 2) * Lx * Lx);
            
            // Actually the standard Matsushima formula:
            // f_limit = 1/(2·dx) · 1/√(1 + (λ·z/L)²)
            // where L = N·dx (physical window size)
            // Hmm, I need to reconsider. The formula from the plan is:
            // fx_limit = 1/(2·dx) · 1/√(1 + (λ·dz / Lx)²)
            // Let me recalculate correctly.

            const fxLimit = fxMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / Lx, 2));
            const fyLimit = fyMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / Ly, 2));

            if (Math.abs(fx) <= fxLimit && Math.abs(fy) <= fyLimit) {
              const phase = kz * dzUm;
              H_re = Math.cos(phase);
              H_im = Math.sin(phase);
            }
            // else: H = 0 (filtered)
          } else {
            const phase = kz * dzUm;
            H_re = Math.cos(phase);
            H_im = Math.sin(phase);
          }
        }
        // else: evanescent → H = (0, 0)

        // Apply H: multiply complex
        const idx = (m * pny + n) * 2;
        const aRe = _padBuf[idx];
        const aIm = _padBuf[idx + 1];
        _padBuf[idx]     = aRe * H_re - aIm * H_im;
        _padBuf[idx + 1] = aRe * H_im + aIm * H_re;
      }
    }

    // IFFT back
    ifftshift(_padBuf, pnx, pny);
    fft2d(_padBuf, pnx, pny, true);
    fftshift(_padBuf, pnx, pny);

    // Extract center region into Uz
    for (let i = 0; i < nx; i++) {
      const srcRow = (ox + i) * pny * 2 + oy * 2;
      const dstRow = i * ny * 2;
      for (let j = 0; j < ny; j++) {
        Uz[dstRow + j * 2]     = _padBuf[srcRow + j * 2];
        Uz[dstRow + j * 2 + 1] = _padBuf[srcRow + j * 2 + 1];
      }
    }

  } else {
    // ─── Non-padded (basic) path ───
    // Copy U0 → work
    for (let i = 0; i < nx * ny * 2; i++) work[i] = U0[i];

    ifftshift(work, nx, ny);
    fft2d(work, nx, ny, false);
    fftshift(work, nx, ny);

    const Lx = nx * dx, Ly = ny * dy;
    const k = 2 * Math.PI / lambdaUm;

    for (let m = 0; m < nx; m++) {
      const fm = m < nx / 2 ? m : m - nx;
      const fx = fm / (nx * dx);
      for (let n = 0; n < ny; n++) {
        const fn = n < ny / 2 ? n : n - ny;
        const fy = fn / (ny * dy);

        const kx = 2 * Math.PI * fx;
        const ky = 2 * Math.PI * fy;
        const kxy2 = kx * kx + ky * ky;

        let H_re = 0, H_im = 0;
        if (kxy2 <= k * k) {
          const kz = Math.sqrt(k * k - kxy2);
          if (bandLimited) {
            const fxMax = 1 / (2 * dx);
            const fyMax = 1 / (2 * dy);
            const fxLimit = fxMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / Lx, 2));
            const fyLimit = fyMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / Ly, 2));
            if (Math.abs(fx) <= fxLimit && Math.abs(fy) <= fyLimit) {
              const phase = kz * dzUm;
              H_re = Math.cos(phase);
              H_im = Math.sin(phase);
            }
          } else {
            const phase = kz * dzUm;
            H_re = Math.cos(phase);
            H_im = Math.sin(phase);
          }
        }
        const idx = (m * ny + n) * 2;
        const aRe = work[idx], aIm = work[idx + 1];
        work[idx]     = aRe * H_re - aIm * H_im;
        work[idx + 1] = aRe * H_im + aIm * H_re;
      }
    }

    ifftshift(work, nx, ny);
    fft2d(work, nx, ny, true);
    fftshift(work, nx, ny);

    // Copy to Uz
    for (let i = 0; i < nx * ny * 2; i++) Uz[i] = work[i];
  }
};
