# Plan: 拆分为独立 Python 计算核心 + 轻量 HTML 展示壳

## 目标

将当前纯浏览器 JS 实现拆成两层：

- **Python 计算核心**：所有物理计算，独立可测试、可脚本调用
- **HTML 展示壳**：只负责 UI 交互和结果展示，不含物理逻辑

---

## 现状分析

当前代码中，物理管线集中在 Worker 内，顺序是：

```
Aperture.generate → Optics.applyTilt → AngSpec.propagate → 强度/相位/截面提取
```

数据流：`Float64Array` 交叉存储 `[Re, Im, ...]`，行主序 `(i * ny + j) * 2`。

这是天然的 Python 迁移边界——只需把 Worker 内的逻辑搬到 Python 函数，HTML 壳通过 HTTP 调用。

---

## 新目录结构

```
/
├── py/                          # Python 计算核心（独立包）
│   ├── __init__.py
│   ├── types.py                 # 数据结构：Field, Params, Result（dataclass）
│   ├── fft.py                   # FFT（复用 scipy.fft，或纯 numpy 自实现）
│   ├── aperture.py              # 光阑生成（原 aperture.js）
│   ├── angspec.py               # 角谱传播（原 angspec.js）
│   ├── optics.py                # 偏轴、相干性（原 optics.js，补完 stub）
│   └── pipeline.py              # 编排：aperture → tilt → propagate → postprocess
│
├── server.py                    # FastAPI 服务（唯一入口）
├── templates/
│   └── index.html.j2            # Jinja2 模板（由 server.py 渲染）
├── static/
│   └── colormap.js              # 保留：主线程渲染，不迁移
│
├── tests/
│   ├── test_fft.py
│   ├── test_aperture.py
│   ├── test_angspec.py
│   └── test_pipeline.py
│
├── js/                          # 迁移后仅剩
│   └── colormap.js              # 主线程查表渲染（纯 UI，不变）
│   └── app_shell.js             # UI 壳：fetch → render，无物理逻辑
│
├── index.html                   # 由 server.py 渲染的入口（或直接指向 server.py 路由）
├── server.mjs                   # 删除
├── js/worker.js                 # 删除
├── js/math/fft.js               # 删除
├── js/physics/                  # 删除
└── CLAUDE.md
```

---

## Python 核心设计

### 数据结构 (`types.py`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    aperture_type: str
    aperture_params: dict
    Nx: int; Ny: int
    dx: float; dy: float      # μm
    lambda_nm: float           # nm
    dz: float                  # mm
    w0: float                  # mm, Gaussian envelope
    tilt_on: bool
    tilt_x_deg: float; tilt_y_deg: float
    temporal_on: bool
    delta_lambda: float        # nm
    M: int                     # wavelength count
    spatial_on: bool
    K: int                     # direction count
    padding: bool = True
    band_limited: bool = True

@dataclass
class Result:
    intensity: np.ndarray      # (Ny, Nx) uint8
    phase: np.ndarray          # (Ny, Nx) uint8
    cross_x: np.ndarray        # (Ny,) float64  — along x-axis
    cross_y: np.ndarray        # (Nx,) float64  — along y-axis
    info: dict                 # fresnel_num, Lx_mm, Ly_mm, ...
```

### 数据布局约定

JS 行主序 `(i * ny + j) * 2` ↔ Python `(j, i)`（列主序/numpy 默认）。

约定：**Python 内部统一使用 `(Ny, Nx)` 即 `(x, y)` 形状**，对外接口清晰，不与 JS 内存布局耦合。

### 管线 (`pipeline.py`)

```python
def compute(params: Params) -> Result:
    field = aperture.generate(params)          # (Ny, Nx) complex64/128
    if params.tilt_on:
        field = optics.apply_tilt(field, params)
    field = angspec.propagate(field, params)   # (Ny, Nx) complex
    return postprocess(field, params)           # Result
```

`postprocess` 提取强度、相位、中心截面，全部用 numpy 向量化，无需显式循环。

### FFT 策略

优先 `scipy.fft`（MKL/FFTW 后端，比纯 JS FFT 快 10-100 倍）。

如果希望零依赖，用 `numpy.fft`（也是 C 实现，足够快）。

**不保留** `js/math/fft.js` 的纯 JS 自实现——Python 有更好的替代。

---

## FastAPI 服务 (`server.py`)

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

app = FastAPI()

# 加载模板
with open("templates/index.html.j2") as f:
    INDEX_TEMPLATE = Template(f.read())

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_TEMPLATE.render()  # 内联 app_shell.js + colormap.js

@app.post("/api/compute")
def api_compute(params: Params):
    result = pipeline.compute(params)
    return JSONResponse({
        "intensity": encode_uint8(result.intensity),
        "phase":     encode_uint8(result.phase),
        "cross_x":   result.cross_x.tolist(),
        "cross_y":   result.cross_y.tolist(),
        "info":      result.info,
    })
```

### 传输格式选择

| 方案 | 优点 | 缺点 |
|------|------|------|
| JSON + base64(array.tobytes()) | 兼容性好，单文件部署 | 33% 体积膨胀 |
| JSON + float list | 人类可读，调试方便 | 256×256 强度 ~196KB JSON |
| protobuf / msgpack | 紧凑 | 需额外依赖 |

**推荐**：先用 JSON + base64 做验证，确认性能后再优化。512×512 的 payload 约 500KB-1MB，在现代网络上 < 100ms。

---

## HTML 壳 (`app_shell.js` + template)

与当前 `app.js` 的区别：

- **删除**：Worker 创建、消息路由、FFT 调用、内存池、所有物理参数解析后的计算触发
- **保留**：UI 双向绑定、Canvas 渲染（强度/相位/截面）、动画、截图、重置
- **新增**：`fetch('/api/compute', {method: 'POST', body: JSON.stringify(params)})`

核心简化：`doCompute()` 从 `worker.postMessage(msg)` 变成：

```js
async function doCompute() {
  const p = gatherParams();
  setStatus('计算中...');
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(p),
  });
  const msg = await res.json();
  handleResult(msg);  // 渲染逻辑不变
}
```

`handleResult` 从 Worker 消息解码改为从 JSON 解码，渲染管线（`applyIntensityLut`、`drawCrossSection`）完全不变。

---

## 迁移顺序

### Phase 1：建立 Python 核心（无服务端）

1. 创建 `py/` 目录，搬移并翻译：
   - `aperture.js` → `py/aperture.py`
   - `angspec.js` → `py/angspec.py`（先平移，再修正 fx/fy bug）
   - `optics.js` → `py/optics.py`（补完 temporalCoherence/spatialCoherence）
   - `fft.js` → `py/fft.py`（用 scipy.fft 替代）
2. 写 `pipeline.py` 编排
3. 写 `py/types.py` 数据结构

**里程碑**：`python -c "from py.pipeline import compute; r = compute(...)"` 能在 notebook/脚本中直接调用。

### Phase 2：测试

为每个模块写 pytest：
- FFT 自测（roundtrip < 1e-12）
- 单缝衍射第一暗纹位置 `x = λz/a`
- 圆孔 Airy 斑第一暗环 `1.22λz/D`
- 与当前 JS 输出做数值对比（dx==dy 时应一致）

**里程碑**：`pytest tests/` 全绿。

### Phase 3：FastAPI 服务

1. 写 `server.py` + `templates/index.html.j2`
2. 搬 `colormap.js` 到 `static/`
3. 写精简 `app_shell.js`（删计算逻辑，换 fetch）

**里程碑**：`python server.py` → 浏览器访问 → UI 正常交互。

### Phase 4：清理

删除 `js/worker.js`、`js/math/`、`js/physics/`、`server.mjs`。

---

## 关键优势

| 方面 | 当前 (纯 JS) | 拆分后 |
|------|-------------|--------|
| 物理逻辑可测试 | ❌ 需 Playwright E2E | ✅ pytest 单元测试 |
| 可脚本调用 | ❌ | ✅ `python -c "from py.pipeline import compute"` |
| 性能 | Worker + 纯 JS FFT，512×512 约百 ms | numpy/scipy C 后端，快 10-100 倍 |
| 并行度 | 单 Worker | 多请求并发（FastAPI + async） |
| 前后端解耦 | ❌ 耦合在 Worker | ✅ HTTP API，可换任意前端 |
| 后续扩展 (D2NN) | 需在 JS 里写 ML | Python 生态（torch/numpy） |

---

## 风险与注意

1. **Base64 传输开销**：512×512 结果约 500KB-1MB，实测延迟。如有需要，可改用二进制 endpoint + `ArrayBuffer` 响应。
2. **并发安全**：`AngSpec.propagate` 里的 `_padBuf` 模块级变量在多请求下不安全。改为函数内局部变量或 session 隔离。
3. **fx/fy bug**：迁移时先修 angspec.js 的轴混淆，再翻译到 Python，避免把 bug 一起搬过去。
