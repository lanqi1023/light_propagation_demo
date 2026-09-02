# 开发任务

## 完成
- **M1**: index.html + style.css — 完整 UI（手风琴侧栏、Canvas 行、仪表盘、控件）
- **M2**: app.js — UI 双向绑定、Worker 创建、requestId 防抖、Canvas putImageData 渲染
- **M2**: worker.js — 消息路由、内存池、完整计算管线（光阑→传播→结果）
- **M3**: colormap.js — Inferno LUT (256×3) + HSV 相位 Alpha 遮罩 (I<5%)
- **M4**: fft.js — 1D/2D FFT、位反转表 + 旋转因子预计算、fftshift/ifftshift、Worker 自测
- **M5**: aperture.js — 单缝/双缝/圆孔/矩形/自由、高斯包络、上传
- **M5**: angspec.js — 基础角谱（频率坐标、倏逝波截止、传递函数）
- **M6**: angspec.js — 2× 零填充（内部 2N×2N，裁剪中心 N×N）
- **M7**: angspec.js — Matsushima 带限 ASM（f_limit = 1/(2·dx) / √(1 + (λ·dz/L)²)）
- **M8**: 图片上传 — File→Canvas→Gray→Float64Array→Worker 缓存
- **M8**: 输入光阑预览 — 主线程 Canvas 渲染输入场
- **M9**: 强度/相位渲染 — Inferno LUT + HSV Alpha 遮罩、putImageData
- **M9**: 截面图 — Canvas 2D、物理坐标 (mm)、轴线标签、图例
- **M10**: 动画扫参 — dz/λ、播放/暂停、~5fps
- **M10**: 截图 — Canvas.toBlob 下载
- **M11**: 偏轴相位 — Optics.applyTilt、接入 Worker 管线
- **测试**: Playwright 27 个测试全部通过（chromium/firefox/webkit）

## 近期修复
- angspec.js: 清理 Matsushima 带限部分的死代码
- worker.js: 简化 samplingOk 计算（消除冗余变量）
- index.html: dz range 步长 0.001 → 1（浏览器卡顿修复）
- app.js: 消除初始加载双重复 requestCompute
- Playwright 配置: webServer + server.mjs 静态文件服务

## 待办（按优先级）

### 高优先级
1. **M12: 时间相干性** — optics.js 实现多波长非相干加权和
   - worker.js `handleCompute` 中检测 `temporalOn` → 遍历 M 个 λ
   - 结果强度 `I = Σ w_m · |U(λ_m)|²`
   - 权重 w_m 为高斯光谱（FWHM = Δλ）
   - `deltaLambdaNum` 范围 0~200nm，`mSelect` 取值 1/3/5/7

2. **M12: 算力保护锁 + 进度条**
   - `gatherParams` 已做网格降级（128/64），配合进度回传
   - Worker 每完成一个子计算 `postMessage({type:'progress', done, total})`
   - 主线程显示进度条（如 "5/25"）

3. **M13: 空间相干性** — X 方向单维角度叠加
   - `Optics.applyTilt` 复用偏轴照明函数
   - K 个方向（高斯权重），结果强度加权和
   - `kSelect` 取值 1/3/5/7

4. **M13: 时间+空间组合模式**
   - 总计算量 = M × K，M×K≥9 降 128，M×K≥25 降 64
   - 进度条 `(m×k / M×K)`

5. **M10: 对比模式** — 理想 vs 当前强度并排 + |ΔI|
   - 第一次计算缓存理想结果，开启非理想后首次计算做差异
   - `comparisonRow` CSS 已隐藏，切换显示

### 中优先级
6. **M14: 物理正确性验证**
   - 夫琅禾费单缝 (sinc² 第一暗纹位置 ÷)
   - 圆孔 Airy 斑 (1.22λz/D)
   - 双缝干涉 (Δx = λz/d)
   - 带限 ASM 远场平滑过渡 + 无混叠
   - 时间退相干条纹对比度下降
   - 空间退相干整体模糊
   - 偏轴平移/形变

7. **M14: 性能调优**
   - DevTools Memory Profiler 确认无泄漏
   - 256×256 实时计算 <100ms

### 低优先级
8. 输入场查看器可缩放
9. 自定义颜色映射
10. 测试覆盖率扩展

## 已知问题
- `crossCanvas`: 800×180 固定分辨率，CSS max-width 但内部不变（非响应式）
- `canvas.toBlob` + 程序化 `a.click()` 下载: Playwright 的 download 事件不可靠（已修改测试规避此问题）
- `dzRange` step=1 导致滑块调整为整数 mm；精确值需通过数字输入框键入
