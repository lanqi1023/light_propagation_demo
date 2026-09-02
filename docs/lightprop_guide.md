# Lightprop 学习指南

## 1. 物理原理

### 1.1 角谱法（Angular Spectrum Method）

角谱法是求解亥姆霍兹方程、计算光场自由空间传播的一种频域方法。

**核心思想**：任意光场都可以分解为不同方向传播的平面波分量（角谱）。每个平面波分量在自由空间传播时只会获得一个相位因子，因此可以在频域处理传播。

**数学表达**：

$$
U(x, y, z) = \iint A(f_x, f_y) \exp\left(i \cdot 2\pi \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} \cdot z\right) \exp\left(i \cdot 2\pi (f_x x + f_y y)\right) \, df_x \, df_y
$$

其中：
- $U(x, y, z)$ 是 $z$ 处的光场
- $A(f_x, f_y)$ 是 $z=0$ 处的角谱（频谱）
- $f_x, f_y$ 是空间频率，单位 $\mu m^{-1}$
- $\lambda$ 是波长

**离散实现**：
1. 对输入光场 $U_0$ 做 2D FFT，得到离散角谱 $A$
2. 乘以传递函数 $H(f_x, f_y) = \exp(i \cdot k_z \cdot dz)$
3. 做逆 FFT 得到输出光场 $U_z$

### 1.2 带限滤波（Matsushima 2009）

**问题**：传递函数 $H(f_x, f_y)$ 在远场（$dz$ 很大）时会变化非常剧烈。如果频率网格不够密，离散采样会导致混叠（aliasing），产生非物理的条纹。

**解决**：在频域施加一个低通滤波器，只保留低频分量。

**截止频率公式**：

$$
f_{\text{limit}} = \frac{1}{2 \cdot dx} \cdot \frac{1}{\sqrt{1 + \left(\dfrac{\lambda \cdot dz}{L}\right)^2}}
$$

其中：
- $dx$ 是像素间距
- $\lambda$ 是波长
- $dz$ 是传播距离
- $L$ 是物理窗口尺寸（$N \cdot dx$）

**物理直觉**：
- $dz \to 0$（近场）：$f_{\text{limit}} \to \frac{1}{2dx}$，不切任何频率
- $dz \to \infty$（远场）：$f_{\text{limit}} \to 0$，几乎所有频率被滤掉
- 波长越长、传播越远，能正确传播的最高频率越低

### 1.3 零填充（Zero-padding）

**问题**：FFT 假设周期性边界条件。如果衍射波传播到窗口边缘，它会从另一侧卷绕回来（wrap-around），造成非物理的伪影。

**解决**：将网格扩大 2 倍（$2N_x \times 2N_y$），将原场放在中心，周围补零。这样衍射波有更多空间扩散，减小边界卷绕误差。

### 1.4 倏逝波截断

**问题**：当空间频率 $|f_x|$ 或 $|f_y|$ 大于 $1/\lambda$ 时，$k_z$ 变为虚数，对应倏逝波（evanescent wave）。倏逝波指数衰减，不参与远场传播。

**解决**：在传递函数中直接将这些频率分量设为 0。

判定条件：

$$
k_{xy}^2 = (2\pi f_x)^2 + (2\pi f_y)^2 \le k^2 = \left(\frac{2\pi}{\lambda}\right)^2
$$

## 2. 代码结构

### 2.1 模块概览

```
lightprop/
├── __init__.py      # 包入口，导出主要 API
├── types.py         # 数据结构：Params（输入参数）、Result（输出结果）
├── aperture.py      # 光阑生成：slit, doubleSlit, circle, rectangle, free, upload
├── fft.py           # 2D FFT 封装：fft2d, fftshift, ifftshift, self_test
├── angspec.py       # 角谱法核心：带限 ASM + 零填充
├── optics.py        # 非理想效应：偏轴、时间相干、空间相干
└── pipeline.py      # 编排器：aperture → tilt → propagate → postprocess
```

### 2.2 数据流

```
Params
  │
  ▼
[aperture.py] U0 = generate(...)          # 生成孔径光场
  │
  ▼
[optics.py]   U0 = apply_tilt(...)        # 可选：偏轴相位
  │
  ▼
[angspec.py]  Uz = propagate(...)         # 角谱法传播
  │
  ▼
[pipeline.py] intensity = |Uz|^2          # 强度
              phase = angle(Uz)            # 相位
              cross_x/y = 中心截面        # 截面图
  │
  ▼
Result
```

### 2.3 关键约定

**数组布局**：
- 形状：`(Nx, Ny)`，complex128
- 行主序：axis 0 = y 方向（rows），axis 1 = x 方向（cols）
- 这与 numpy 默认一致：`(row, col) = (y, x)`

**坐标系**：
- `xs`：x 坐标数组，形状 `(Ny,)` 或 `(Nx, Ny)`，随列索引变化
- `ys`：y 坐标数组，形状 `(Nx,)` 或 `(Nx, Ny)`，随行索引变化
- 原点 $(0, 0)$ 位于网格中心

**单位**：
- `dx, dy`：$\mu m$（微米）
- `lambda_nm`：nm（纳米），内部转换为 $\mu m$
- `dz`：mm（毫米），内部转换为 $\mu m$
- `w0`：mm（毫米），高斯束腰半径

## 3. 关键参数说明

### 3.1 网格参数

- `Nx, Ny`：网格尺寸（行数，列数）
- `dx, dy`：像素间距（$\mu m$）
- 物理窗口：$L_x = N_x \cdot dx$，$L_y = N_y \cdot dy$（$\mu m$）

**选择建议**：
- 实时模式：256×256 或 512×512
- 非理想模式（相干性）：自动降级到 128×128 或 64×64
- 远场传播：需要更大的网格或更小的 dx

### 3.2 光阑参数

- `aperture_type`：光阑类型
- `aperture_params`：光阑参数（宽度、半径等）

### 3.3 传播参数

- `dz`：传播距离（mm）
- `padding`：是否零填充（推荐 True）
- `band_limited`：是否带限滤波（推荐 True）

### 3.4 非理想效应参数

- `tilt_on`：偏轴照明
- `temporal_on`：时间相干性（宽光谱）
- `spatial_on`：空间相干性（扩展光源）

## 4. 常见问题

### 4.1 输出图样方向反了？

检查以下几点：
1. `aperture.py` 中的 `xs, ys` 坐标是否正确
2. `angspec.py` 中的 `fx, fy` 频率坐标是否正确
3. `pipeline.py` 中的 `cross_x, cross_y` 提取是否正确
4. 前端 `app_shell.js` 中的 canvas 宽高是否对应

**验证方法**：运行 `verify_end_to_end.py`，检查 aperture 和 propagation 的输出。

### 4.2 远场输出全平？

可能是带限滤波过度。尝试：
1. 增大网格尺寸（Nx, Ny）
2. 减小像素间距（dx, dy）
3. 关闭带限滤波（`band_limited=False`）看是否是滤波导致

### 4.3 计算很慢？

- 256×256 网格：~20-30 ms（CPU）
- 512×512 网格：~100 ms（CPU）
- 如果开启相干性（$M \times K$ 次传播），计算量会增加 $M \times K$ 倍

## 5. 验证方法

### 5.1 运行测试

```bash
pytest tests/ -v
```

### 5.2 手动验证

```bash
python verify_end_to_end.py
```

### 5.3 启动服务

```bash
python server.py
# 访问 http://localhost:8080
```

## 6. 参考资料

1. Matsushima, K., & Shimobaba, T. (2009). "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." *Optics Express*, 17(22), 19662-19673.
2. Voelz, D. G. (2011). *Computational Fourier Optics: A MATLAB Tutorial*. SPIE Press.
3. Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.). Roberts & Company.
