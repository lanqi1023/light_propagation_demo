// Inferno-like colormap: 256 × 3 uint8 lookup table
// Key points (t, R, G, B) in [0,1] — piecewise linear interpolation
const COLORMAP_KEYS = [
  [0,   0.000, 0.000, 0.016],
  [0.1, 0.023, 0.011, 0.172],
  [0.2, 0.106, 0.014, 0.366],
  [0.3, 0.238, 0.030, 0.476],
  [0.4, 0.388, 0.074, 0.501],
  [0.5, 0.545, 0.142, 0.466],
  [0.6, 0.697, 0.232, 0.386],
  [0.7, 0.828, 0.347, 0.276],
  [0.8, 0.925, 0.501, 0.144],
  [0.9, 0.974, 0.683, 0.036],
  [1,   1.000, 0.882, 0.098],
];

const LUT_SIZE = 256;

// intensityLut: Uint8Array[256 * 3], R0,G0,B0, R1,G1,B1, ...
const intensityLut = new Uint8Array(LUT_SIZE * 3);

(function buildIntensityLut() {
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    // find surrounding keypoints
    let lo = 0;
    for (let k = 0; k < COLORMAP_KEYS.length - 1; k++) {
      if (t >= COLORMAP_KEYS[k][0] && t <= COLORMAP_KEYS[k + 1][0]) {
        lo = k;
        break;
      }
    }
    const k0 = COLORMAP_KEYS[lo];
    const k1 = COLORMAP_KEYS[lo + 1];
    const f = (t - k0[0]) / (k1[0] - k0[0] || 1);
    const r = Math.round(255 * (k0[1] + f * (k1[1] - k0[1])));
    const g = Math.round(255 * (k0[2] + f * (k1[2] - k0[2])));
    const b = Math.round(255 * (k0[3] + f * (k1[3] - k0[3])));
    const idx = i * 3;
    intensityLut[idx] = Math.min(255, Math.max(0, r));
    intensityLut[idx + 1] = Math.min(255, Math.max(0, g));
    intensityLut[idx + 2] = Math.min(255, Math.max(0, b));
  }
})();

// Render intensity array (Uint8Array, 0-255) to Uint8ClampedArray RGBA
function applyIntensityLut(src, dst) {
  const len = src.length;
  for (let i = 0; i < len; i++) {
    const v = src[i];
    const lutIdx = v * 3;
    const pixelIdx = i * 4;
    dst[pixelIdx]     = intensityLut[lutIdx];
    dst[pixelIdx + 1] = intensityLut[lutIdx + 1];
    dst[pixelIdx + 2] = intensityLut[lutIdx + 2];
    dst[pixelIdx + 3] = 255;
  }
}

// Render phase array (Uint8Array, 0-255 maps to -π..π) + intensity mask
// intensityRef: Uint8Array (same size), used for alpha mask
function applyPhaseLut(src, intensityRef, dst, alphaThreshold) {
  const len = src.length;
  alphaThreshold = alphaThreshold || 5; // 5/255 threshold
  for (let i = 0; i < len; i++) {
    const phase01 = src[i] / 255;          // [0, 1]
    const hue = phase01 * 360;             // [0, 360]
    const s = 1, v = 1;
    // HSV → RGB
    const h = hue / 60;
    const hi = Math.floor(h) % 6;
    const f = h - Math.floor(h);
    const p = v * (1 - s);
    const q = v * (1 - s * f);
    const t = v * (1 - s * (1 - f));
    let r, g, b;
    switch (hi) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
    const pixelIdx = i * 4;
    dst[pixelIdx]     = Math.round(r * 255);
    dst[pixelIdx + 1] = Math.round(g * 255);
    dst[pixelIdx + 2] = Math.round(b * 255);
    dst[pixelIdx + 3] = (intensityRef[i] < alphaThreshold) ? 0 : 255;
  }
}
