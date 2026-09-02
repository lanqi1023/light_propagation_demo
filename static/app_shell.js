// ─── Main Thread Shell (physics logic moved to Python backend) ───
// Version: python-backend-only
(function() {
  'use strict';

  const $ = id => document.getElementById(id);
  const $$ = sel => document.querySelectorAll(sel);

  function safeParseInt(v, fallback) { const n = parseInt(v); return isNaN(n) ? fallback : n; }
  function safeParseFloat(v, fallback) { const n = parseFloat(v); return isNaN(n) ? fallback : n; }

  // Canvas + context
  const inputCanvas   = $('inputCanvas');
  const intensityCanvas = $('intensityCanvas');
  const phaseCanvas   = $('phaseCanvas');
  const crossCanvas   = $('crossCanvas');
  const idealCanvas   = $('idealCanvas');
  const currentCanvas = $('currentCanvas');
  const diffCanvas    = $('diffCanvas');

  const ctxInput     = inputCanvas.getContext('2d');
  const ctxIntensity = intensityCanvas.getContext('2d');
  const ctxPhase     = phaseCanvas.getContext('2d');
  const ctxCross     = crossCanvas.getContext('2d');
  const ctxIdeal     = idealCanvas.getContext('2d');
  const ctxCurrent   = currentCanvas.getContext('2d');
  const ctxDiff      = diffCanvas.getContext('2d');

  const CANVAS_SIZE = 256;
  let rgbaIntensity = new Uint8ClampedArray(CANVAS_SIZE * CANVAS_SIZE * 4);
  let rgbaPhase     = new Uint8ClampedArray(CANVAS_SIZE * CANVAS_SIZE * 4);
  let imageDataIntensity = new ImageData(rgbaIntensity, CANVAS_SIZE, CANVAS_SIZE);
  let imageDataPhase     = new ImageData(rgbaPhase, CANVAS_SIZE, CANVAS_SIZE);

  // ─── State ───
  let debounceTimer = null;
  let isComputing = false;
  let pendingRecompute = false;
  let isHighQuality = false;
  let lastResult = null;
  let uploadedPixels = null;

  // ─── Decode base64 from Python backend ───
  function decodeBase64(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  // ─── Handle computation result ───
  function handleResult(msg) {
    if (pendingRecompute) {
      pendingRecompute = false;
      requestCompute();
      return;
    }
    isComputing = false;
    lastResult = msg;

    const Nx = msg.info.nx || CANVAS_SIZE;  // rows
    const Ny = msg.info.ny || CANVAS_SIZE;  // cols
    const nPixels = Nx * Ny;

    // Decode base64 payloads
    const intensity = decodeBase64(msg.intensity);
    const phase = decodeBase64(msg.phase);
    const crossX = new Float64Array(msg.cross_x);
    const crossY = new Float64Array(msg.cross_y);

    // Resize result canvases: width=cols=x, height=rows=y
    [intensityCanvas, phaseCanvas].forEach(c => {
      if (c.width !== Ny || c.height !== Nx) { c.width = Ny; c.height = Nx; }
    });

    // Re-allocate pixel buffers if size changed
    if (!imageDataIntensity || imageDataIntensity.width !== Ny || imageDataIntensity.height !== Nx) {
      rgbaIntensity = new Uint8ClampedArray(nPixels * 4);
      rgbaPhase     = new Uint8ClampedArray(nPixels * 4);
      imageDataIntensity = new ImageData(rgbaIntensity, Ny, Nx);
      imageDataPhase     = new ImageData(rgbaPhase, Ny, Nx);
    }

    // Render intensity
    applyIntensityLut(intensity, rgbaIntensity);
    ctxIntensity.putImageData(imageDataIntensity, 0, 0);

    // Render phase (with alpha mask)
    applyPhaseLut(phase, intensity, rgbaPhase, 5);
    ctxPhase.putImageData(imageDataPhase, 0, 0);

    // Render cross-section
    drawCrossSection(crossX, crossY, msg.info);

    // Update dashboard
    updateDashboard(msg.info);

    setStatus('✓ 完成 (' + msg.info.computationTime + 'ms)');
  }

  // ─── Draw cross-section (unchanged from original) ───
  function drawCrossSection(crossX, crossY, info) {
    const w = crossCanvas.width;
    const h = crossCanvas.height;
    const ctx = ctxCross;
    const pad = { top: 15, bottom: 25, left: 50, right: 20 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    let maxVal = 0;
    for (let k = 0; k < crossX.length; k++) if (crossX[k] > maxVal) maxVal = crossX[k];
    for (let k = 0; k < crossY.length; k++) if (crossY[k] > maxVal) maxVal = crossY[k];
    if (maxVal === 0) maxVal = 1;

    function toPixelX(val, min, max) {
      return pad.left + (val - min) / (max - min) * plotW;
    }
    function toPixelY(val) {
      return pad.top + plotH - (val / maxVal) * plotH;
    }

    const Lx = info.Lx || 1;
    const Ly = info.Ly || 1;

    // crossX (from Python): varies along x-direction → use Lx
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const ny = crossX.length;
    for (let j = 0; j < ny; j++) {
      const physX = (j / (ny - 1) - 0.5) * Lx;
      const px = toPixelX(physX, -Lx / 2, Lx / 2);
      const py = toPixelY(crossX[j]);
      j === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // crossY (from Python): varies along y-direction → use Ly
    ctx.strokeStyle = '#f0883e';
    ctx.beginPath();
    const nx = crossY.length;
    for (let i = 0; i < nx; i++) {
      const physY = (i / (nx - 1) - 0.5) * Ly;
      const px = toPixelX(physY, -Ly / 2, Ly / 2);
      const py = toPixelY(crossY[i]);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Axes
    ctx.strokeStyle = '#484f58';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = 5;
    for (let k = 0; k <= xTicks; k++) {
      const phys = -Lx / 2 + (k / xTicks) * Lx;
      const px = pad.left + (k / xTicks) * plotW;
      ctx.fillText(phys.toFixed(1), px, pad.top + plotH + 4);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yTicks = 4;
    for (let k = 0; k <= yTicks; k++) {
      const val = (k / yTicks) * maxVal;
      const py = pad.top + plotH - (k / yTicks) * plotH;
      ctx.fillText(val.toFixed(1), pad.left - 4, py);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#484f58';
    ctx.font = '10px sans-serif';
    ctx.fillText('位置 (mm)', pad.left + plotW / 2, pad.top + plotH + 16);
    ctx.save();
    ctx.translate(10, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('强度', 0, 0);
    ctx.restore();

    // Legend
    ctx.fillStyle = '#58a6ff';
    ctx.fillRect(pad.left + plotW - 120, pad.top + 4, 12, 3);
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('X 截面', pad.left + plotW - 104, pad.top);
    ctx.fillStyle = '#f0883e';
    ctx.fillRect(pad.left + plotW - 120, pad.top + 16, 12, 3);
    ctx.fillStyle = '#8b949e';
    ctx.fillText('Y 截面', pad.left + plotW - 104, pad.top + 12);
  }

  // ─── Update dashboard (unchanged) ───
  function updateDashboard(info) {
    $('fresnelDisplay').textContent = info.fresnelNum ? info.fresnelNum.toFixed(2) : '—';
    $('dashLx').textContent = info.Lx ? info.Lx.toFixed(3) + ' mm' : '—';
    $('dashLy').textContent = info.Ly ? info.Ly.toFixed(3) + ' mm' : '—';
    $('timeDisplay').textContent = info.computationTime ? info.computationTime + ' ms' : '—';
    $('dashN').textContent = (info.ny || '?') + '×' + (info.nx || '?');

    const statusEl = $('samplingStatus');
    if (info.samplingOk) {
      statusEl.textContent = '✓ 充足';
      statusEl.className = 'dash-value ok';
    } else {
      statusEl.textContent = '⚠ 不足';
      statusEl.className = 'dash-value warn';
    }
  }

  // ─── Gather all params from UI (unchanged) ───
  function gatherParams() {
    const apertureType = document.querySelector('input[name="aperture"]:checked').value;
    const params = {};

    switch (apertureType) {
      case 'slit':
        params.width = parseFloat($('slitWidthNum').value);
        break;
      case 'doubleSlit':
        params.width = parseFloat($('dsWidthNum').value);
        params.separation = parseFloat($('dsSepNum').value);
        break;
      case 'circle':
        params.radius = parseFloat($('circleRadiusNum').value);
        break;
      case 'rectangle':
        params.width = parseFloat($('rectWidthNum').value);
        params.height = parseFloat($('rectHeightNum').value);
        break;
      case 'upload':
      case 'free':
        break;
    }

    let Nx = safeParseInt($('nxSelect').value, 256);
    let Ny = safeParseInt($('nySelect').value, 256);
    const dx = safeParseFloat($('dxNum').value, 10);
    const dy = safeParseFloat($('dyNum').value, 10);
    const lambda_nm = safeParseFloat($('lambdaNum').value, 532);
    const dz = safeParseFloat($('dzNum').value, 100);
    const w0 = safeParseFloat($('w0Num').value, 100);

    const tiltOn = $('tiltEnable').checked;
    const temporalOn = $('temporalCoherenceEnable').checked;
    const spatialOn = $('spatialCoherenceEnable').checked;

    if (temporalOn || spatialOn) {
      if (Nx > 128) Nx = 128;
      if (Ny > 128) Ny = 128;
    }
    if (temporalOn && spatialOn) {
      if (Nx > 64) Nx = 64;
      if (Ny > 64) Ny = 64;
    }
    if (!isHighQuality && !temporalOn && !spatialOn) {
      if (Nx > 256) Nx = 256;
      if (Ny > 256) Ny = 256;
    }

    return {
      aperture_type: apertureType,
      aperture_params: params,
      Nx, Ny, dx, dy,
      lambda_nm, dz, w0,
      tilt_on: tiltOn,
      tilt_x_deg: tiltOn ? parseFloat($('tiltXNum').value) : 0,
      tilt_y_deg: tiltOn ? parseFloat($('tiltYNum').value) : 0,
      temporal_on: temporalOn,
      delta_lambda: temporalOn ? parseFloat($('deltaLambdaNum').value) : 0,
      M: temporalOn ? parseInt($('mSelect').value) : 1,
      spatial_on: spatialOn,
      K: spatialOn ? parseInt($('kSelect').value) : 1,
      padding: true,
      band_limited: true,
    };
  }

  // ─── Trigger recompute via HTTP ───
  function requestCompute() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      doCompute();
    }, 120);
  }

  async function doCompute() {
    if (isComputing) {
      pendingRecompute = true;
      return;
    }
    const p = gatherParams();
    isComputing = true;
    setStatus('计算中... (' + p.Nx + '×' + p.Ny + ')');

    // Render input preview (unchanged)
    renderInputPreview(p);

    // Update Lx/Ly display
    const Lx = p.Nx * p.dx / 1000;
    const Ly = p.Ny * p.dy / 1000;
    $('lxDisplay').textContent = Lx.toFixed(3);
    $('lyDisplay').textContent = Ly.toFixed(3);

    try {
      const res = await fetch('/api/compute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: p }),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const msg = await res.json();
      handleResult(msg);
    } catch (err) {
      console.error('[Shell] compute failed:', err);
      setStatus('错误: ' + err.message);
      isComputing = false;
    }
  }

  // ─── Render input preview (unchanged) ───
  function renderInputPreview(p) {
    const nx = p.Nx, ny = p.Ny;
    const canvas = inputCanvas;
    const ctx = ctxInput;
    canvas.width = Math.min(256, nx);
    canvas.height = Math.min(256, ny);

    const imgData = ctx.createImageData(canvas.width, canvas.height);
    const data = imgData.data;

    if (p.aperture_type === 'upload' && uploadedPixels) {
      const maxIdx = uploadedPixels.length;
      for (let i = 0; i < canvas.width * canvas.height; i++) {
        const v = i < maxIdx ? Math.min(255, Math.max(0, Math.round(uploadedPixels[i] * 255))) : 0;
        const idx = i * 4;
        data[idx] = data[idx + 1] = data[idx + 2] = v;
        data[idx + 3] = 255;
      }
    } else {
      const xs = new Float64Array(ny);
      const ys = new Float64Array(nx);
      const xc = (nx - 1) / 2, yc = (ny - 1) / 2;
      for (let i = 0; i < nx; i++) ys[i] = (i - yc) * p.dy;
      for (let j = 0; j < ny; j++) xs[j] = (j - xc) * p.dx;

      const scaleX = canvas.width / nx;
      const scaleY = canvas.height / ny;

      for (let i = 0; i < canvas.width; i++) {
        for (let j = 0; j < canvas.height; j++) {
          const ni = Math.floor(i / scaleX);
          const nj = Math.floor(j / scaleY);
          let val = 0;

          switch (p.aperture_type) {
            case 'slit':
              val = Math.abs(xs[nj]) <= p.aperture_params.width / 2 ? 1 : 0;
              break;
            case 'doubleSlit': {
              const halfW = p.aperture_params.width / 2;
              const halfS = p.aperture_params.separation / 2;
              val = (Math.abs(xs[nj] - halfS) <= halfW || Math.abs(xs[nj] + halfS) <= halfW) ? 1 : 0;
              break;
            }
            case 'circle':
              val = (xs[nj] * xs[nj] + ys[ni] * ys[ni] <= p.aperture_params.radius * p.aperture_params.radius) ? 1 : 0;
              break;
            case 'rectangle':
              val = (Math.abs(xs[nj]) <= p.aperture_params.width / 2 && Math.abs(ys[ni]) <= p.aperture_params.height / 2) ? 1 : 0;
              break;
            case 'free':
              val = 1;
              break;
          }

          const v = Math.round(val * 255);
          const idx = (j * canvas.width + i) * 4;
          data[idx] = data[idx + 1] = data[idx + 2] = v;
          data[idx + 3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // ─── Set status ───
  function setStatus(msg) {
    $('statusMessage').textContent = msg;
  }

  // ─── UI Bindings (unchanged) ───
  function bindSlider(sliderId, numId, onChange) {
    const slider = $(sliderId);
    const num = $(numId);
    if (!slider || !num) return;
    slider.addEventListener('input', function() {
      num.value = slider.value;
      if (onChange) onChange();
      else requestCompute();
    });
    num.addEventListener('change', function() {
      let v = parseFloat(num.value);
      const min = parseFloat(num.min);
      const max = parseFloat(num.max);
      if (v < min) v = min;
      if (v > max) v = max;
      num.value = v;
      slider.value = v;
      if (onChange) onChange();
      else requestCompute();
    });
  }

  function bindSelect(id, onChange) {
    const el = $(id);
    if (!el) return;
    el.addEventListener('change', () => {
      if (onChange) onChange();
      requestCompute();
    });
  }

  function bindCheckbox(id, showId, onChange) {
    const cb = $(id);
    const target = $(showId);
    if (!cb) return;
    cb.addEventListener('change', function() {
      if (target) target.style.display = cb.checked ? 'block' : 'none';
      if (onChange) onChange();
      requestCompute();
    });
  }

  function initUI() {
    bindSlider('dxRange', 'dxNum');
    bindSlider('dyRange', 'dyNum');
    bindSlider('lambdaRange', 'lambdaNum');
    bindSlider('dzRange', 'dzNum');
    bindSlider('w0Range', 'w0Num');
    bindSlider('slitWidthRange', 'slitWidthNum');
    bindSlider('dsWidthRange', 'dsWidthNum');
    bindSlider('dsSepRange', 'dsSepNum');
    bindSlider('circleRadiusRange', 'circleRadiusNum');
    bindSlider('rectWidthRange', 'rectWidthNum');
    bindSlider('rectHeightRange', 'rectHeightNum');
    bindSlider('tiltXRange', 'tiltXNum');
    bindSlider('tiltYRange', 'tiltYNum');
    bindSlider('deltaLambdaRange', 'deltaLambdaNum');

    bindSelect('nxSelect', () => updateLxLy());
    bindSelect('nySelect', () => updateLxLy());
    bindSelect('mSelect');
    bindSelect('kSelect');
    bindSelect('sweepVar');

    bindCheckbox('tiltEnable', 'tiltParams');
    bindCheckbox('temporalCoherenceEnable', 'temporalParams');
    bindCheckbox('spatialCoherenceEnable', 'spatialParams');

    $$('input[name="aperture"]').forEach(radio => {
      radio.addEventListener('change', function() {
        showApertureParams(this.value);
        requestCompute();
      });
    });

    $$('.group-title').forEach(el => {
      el.addEventListener('click', function() {
        const target = $(this.dataset.target);
        if (target) {
          target.style.display = target.style.display === 'none' ? '' : 'none';
        }
      });
    });

    function toggleMode(target) {
      if (target) {
        isHighQuality = target.id === 'modeHQ';
      } else {
        isHighQuality = !isHighQuality;
      }
      $('modeReal').className = 'mode-option' + (isHighQuality ? '' : ' active');
      $('modeHQ').className   = 'mode-option' + (isHighQuality ? ' active' : '');
      const oldNx = safeParseInt($('nxSelect').value, 256);
      const newNx = isHighQuality ? 512 : 256;
      const scale = oldNx / newNx;
      const dx = safeParseFloat($('dxNum').value, 10);
      const dy = safeParseFloat($('dyNum').value, 10);
      ['dx', 'dy'].forEach(axis => {
        const newVal = axis === 'dx' ? dx * scale : dy * scale;
        const clamped = Math.max(1, Math.min(50, Math.round(newVal * 100) / 100));
        $(axis + 'Range').value = clamped;
        $(axis + 'Num').value = clamped;
      });
      $('nxSelect').value = newNx;
      $('nySelect').value = newNx;
      updateLxLy();
      requestCompute();
    }
    $('modeReal').addEventListener('click', function() { toggleMode(this); });
    $('modeHQ').addEventListener('click', function() { toggleMode(this); });

    $('imageUpload').addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(ev) {
        const img = new Image();
        img.onload = function() {
          const p = gatherParams();
          const tmpCanvas = document.createElement('canvas');
          tmpCanvas.width = p.Nx;
          tmpCanvas.height = p.Ny;
          const tmpCtx = tmpCanvas.getContext('2d');
          tmpCtx.drawImage(img, 0, 0, p.Nx, p.Ny);
          const imgData = tmpCtx.getImageData(0, 0, p.Nx, p.Ny);
          const pixels = new Float64Array(p.Nx * p.Ny);
          for (let k = 0; k < p.Nx * p.Ny; k++) {
            pixels[k] = (imgData.data[k * 4] + imgData.data[k * 4 + 1] + imgData.data[k * 4 + 2]) / (3 * 255);
          }
          uploadedPixels = new Float64Array(pixels);
          requestCompute();
        };
        img.onerror = function() { setStatus('图片加载失败'); };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    });

    $('playBtn').addEventListener('click', toggleAnimation);
    $('screenshotBtn').addEventListener('click', takeScreenshot);
    $('resetBtn').addEventListener('click', resetAll);

    updateLxLy();
    showApertureParams('slit');
  }

  function showApertureParams(type) {
    const groups = ['slitParams', 'doubleSlitParams', 'circleParams', 'rectangleParams', 'uploadParams'];
    groups.forEach(id => { $(id).style.display = 'none'; });
    switch (type) {
      case 'slit': $('slitParams').style.display = ''; break;
      case 'doubleSlit': $('doubleSlitParams').style.display = ''; break;
      case 'circle': $('circleParams').style.display = ''; break;
      case 'rectangle': $('rectangleParams').style.display = ''; break;
      case 'upload': $('uploadParams').style.display = ''; break;
    }
  }

  function updateLxLy() {
    const Nx = safeParseInt($('nxSelect').value, 256);
    const Ny = safeParseInt($('nySelect').value, 256);
    const dx = safeParseFloat($('dxNum').value, 10);
    const dy = safeParseFloat($('dyNum').value, 10);
    $('lxDisplay').textContent = (Nx * dx / 1000).toFixed(3);
    $('lyDisplay').textContent = (Ny * dy / 1000).toFixed(3);
  }

  // ─── Animation (unchanged) ───
  let animId = null;
  let animRunning = false;
  function toggleAnimation() {
    if (animRunning) {
      animRunning = false;
      if (animId) cancelAnimationFrame(animId);
      $('playBtn').textContent = '▶';
      setStatus('就绪');
      return;
    }
    animRunning = true;
    $('playBtn').textContent = '⏹';
    runAnimation();
  }

  function runAnimation() {
    if (!animRunning) return;
    const sweepVar = $('sweepVar').value;
    const t = Date.now() / 2000;
    if (sweepVar === 'dz') {
      const min = 1, max = 500;
      const val = min * Math.pow(max / min, (Math.sin(t) + 1) / 2);
      $('dzNum').value = val.toFixed(3);
      $('dzRange').value = val;
    } else {
      const min = 400, max = 700;
      const val = min + (max - min) * (Math.sin(t) + 1) / 2;
      $('lambdaNum').value = Math.round(val);
      $('lambdaRange').value = Math.round(val);
    }
    doCompute();
    animId = setTimeout(() => runAnimation(), 200);
  }

  // ─── Screenshot (unchanged) ───
  function takeScreenshot() {
    const canvas = intensityCanvas;
    canvas.toBlob(function(blob) {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'intensity_' + Date.now() + '.png';
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ─── Reset (unchanged) ───
  function resetAll() {
    $('lambdaNum').value = 532; $('lambdaRange').value = 532;
    $('dzNum').value = 100; $('dzRange').value = 100;
    $('dxNum').value = 10; $('dxRange').value = 10;
    $('dyNum').value = 10; $('dyRange').value = 10;
    $('nxSelect').value = 256;
    $('nySelect').value = 256;
    $('w0Num').value = 100; $('w0Range').value = 100;
    $('tiltEnable').checked = false;
    $('tiltParams').style.display = 'none';
    $('tiltXNum').value = 0; $('tiltXRange').value = 0;
    $('tiltYNum').value = 0; $('tiltYRange').value = 0;
    $('temporalCoherenceEnable').checked = false;
    $('temporalParams').style.display = 'none';
    $('deltaLambdaNum').value = 20; $('deltaLambdaRange').value = 20;
    $('mSelect').value = 5;
    $('spatialCoherenceEnable').checked = false;
    $('spatialParams').style.display = 'none';
    $('kSelect').value = 5;
    $('comparisonRow').style.display = 'none';
    uploadedPixels = null;
    isHighQuality = false;
    $('modeReal').className = 'mode-option active';
    $('modeHQ').className = 'mode-option';
    if (animRunning) toggleAnimation();
    requestCompute();
  }

  // ─── Boot ───
  initUI();
  setTimeout(requestCompute, 300);
})();
