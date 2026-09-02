# 角谱法中的 Matsushima 带限

## 1. 问题：角谱法的数值混叠

角谱法在频域中计算光传播，其传递函数为：

$$
H(f_x, f_y) = \exp\left(j \cdot k_z \cdot \Delta z\right),
\qquad
k_z = \sqrt{k^2 - (2\pi f_x)^2 - (2\pi f_y)^2},
\qquad
k = \frac{2\pi}{\lambda}
$$

这个函数本身在数学上是精确的——它直接来自亥姆霍兹方程的通解。问题出在**数值离散化**上。

当我们用 FFT 计算时，频率空间被离散为 $N$ 个点，间距 $\Delta f = 1/(N \cdot \Delta x)$。传递函数 $H(f_x, f_y)$ 在这些离散点上求值。但 $k_z$ 在 $|f_x| \to 1/(2\Delta x)$ 附近的导数趋于无穷大：

$$
\frac{\partial k_z}{\partial f_x} = \frac{-k_x}{k_z} = \frac{-2\pi f_x}{\sqrt{k^2 - (2\pi f_x)^2 - (2\pi f_y)^2}}
$$

当 $|f_x|$ 接近奈奎斯特频率 $1/(2\Delta x)$ 时，分母趋于零，$k_z$ 的相位变化率发散。离散采样无法正确表示这种快速振荡的函数——这就是欠采样，必然导致**混叠（aliasing）**。

**混叠在图像上的表现**：
- 衍射图案出现非物理的条纹或纹理
- 远场传播时能量"反弹"回窗口另一侧
- 关掉带限截屏（`screenshots/nobandlimit.png`）可以看到图像被高频噪声淹没，PNG 体积从 127KB 膨胀到 266KB

---

## 2. 解决方案：Matsushima 2009 带限滤波器

Matsushima 在 2009 年提出一个简洁的解析截止频率公式（Matsushima, K., "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Optics Express*, 2009）：

$$
f_{\text{limit}} = \frac{1}{2\Delta x} \cdot \frac{1}{\sqrt{1 + \left(\dfrac{\lambda \cdot \Delta z}{L}\right)^2}}
$$

其中 $L = N \cdot \Delta x$ 是计算窗口的物理尺寸。

### 物理直觉

- 当 $\Delta z \to 0$（近场）：$f_{\text{limit}} \to 1/(2\Delta x)$，不切任何频率
- 当 $\Delta z \to \infty$（远场）：$f_{\text{limit}} \to 0$，绝大部分频率被滤掉
- 波长越长、传播越远，能正确传播的最高频率越低——符合物理预期

### 在代码中的实现

```js
// angspec.js 第 85-96 行
const fxMax = 1 / (2 * dx);         // 奈奎斯特频率
const fyMax = 1 / (2 * dy);
const fxLimit = fxMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / Lx, 2));
const fyLimit = fyMax / Math.sqrt(1 + Math.pow(lambdaUm * dzUm / Ly, 2));

if (Math.abs(fx) <= fxLimit && Math.abs(fy) <= fyLimit) {
  const phase = kz * dzUm;
  H_re = Math.cos(phase);
  H_im = Math.sin(phase);
}
// else: H = (0, 0) —— 超出带限的频率被直接滤除
```

注意这里的 `lambdaUm` 单位是 μm（$\lambda_{\text{nm}} / 1000$），`dzUm` 单位是 μm（$dz_{\text{mm}} \times 1000$），`Lx = nx \cdot dx$ 单位是 μm。分子分母单位一致，无量纲比值 $(\lambda \cdot \Delta z / L)^2$ 正确。

### 滤波效果

当 $|f_x| > f_{\text{limit}}$ 时，传递函数被强置为零。这意味着这些频率对应的角谱分量被完全阻止传播。**这些分量即使满足传播条件（$k_{xy}^2 \leq k^2$），也因为数值欠采样而被滤除**——这是带限和倏逝波截断的本质区别。

---

## 3. 与倏逝波截断的区别

| 特性 | 倏逝波截断 | Matsushima 带限 |
|------|-----------|----------------|
| **判据** | $k_{xy}^2 > k^2$ | $\|f_x\| > f_{\text{limit}}$ |
| **物理依据** | $k_z$ 变为纯虚数，波指数衰减 | 欠采样的传递函数会产生混叠 |
| **本质** | 物理滤波 | 数值保护 |
| **被滤频率的位置** | 靠近频谱角落（极高 $f$） | 中等频率也可能被滤（远场时） |
| **$\Delta z \to 0$ 行为** | 不滤波（除非倏逝波） | 不滤波（$f_{\text{limit}} \to f_{\text{max}}$） |
| **$\Delta z \to \infty$ 行为** | 不发生变化 | 几乎全部频率被滤 |

在代码中两者是串行关系：

```js
if (kxy2 <= k * k) {           // 第一步：通过倏逝波截断
  // ... propagating modes ...
  if (bandLimited) {           // 第二步：通过带限检查
    if (Math.abs(fx) <= fxLimit && Math.abs(fy) <= fyLimit) {
      H = exp(j * kz * dz);    // 最终通过
    } // else: H = 0 (filtered)
  } else {
    H = exp(j * kz * dz);      // 无带限时直接通过
  }
} // else: H = 0 (evanescent)
```

---

## 4. 与零填充的区别

零填充和带限是两个不同层面的 anti-aliasing 技术：

| 特性 | 零填充 (Zero-padding) | 带限 (Band-limiting) |
|------|----------------------|---------------------|
| **域** | 空间域 | 频域 |
| **做法** | 2N×2N 网格，原场放中央，周围补零 | 频域中对传递函数施加低通滤波 |
| **解决的问题** | 衍射波扩散到窗口边缘后从另一侧卷绕（周期边界伪影） | 传递函数欠采样导致的混叠 |

两者可以独立开关。对应截图 `screenshots/`：

| 截图 | padding | bandlimit | 现象 |
|------|---------|-----------|------|
| `padding+bandlimit.png` | ✅ | ✅ | 正确结果，双缝干涉条纹清晰 |
| `nopadding.png` | ❌ | ✅ | 仅边缘有些许卷绕（带限抑制了高频，卷绕不明显） |
| `nobandlimit.png` | ✅ | ❌ | 虽无卷绕，但高频噪声充斥全图，条纹被污染 |
| `nopadding+nobandlimit.png` | ❌ | ❌ | 混叠 + 卷绕叠加，结果完全不可用 |

---

## 5. 单元一致性说明

代码中三个关键物理量的单位：

| 变量 | 符号 | 单位 |
|------|------|------|
| `lambda` | $\lambda$ | nm（输入），代码内转为 μm：`lambdaUm = lambda / 1000` |
| `dx`, `dy` | $\Delta x, \Delta y$ | μm |
| `dz` | $\Delta z$ | mm（输入），代码内转为 μm：`dzUm = dz * 1000` |
| `Lx` | $L_x = N_x \cdot \Delta x$ | μm |
| `fxLimit` | $f_{\text{limit}}$ | μm$^{-1}$ |

频率 $f_x = m / (N_x \cdot \Delta x)$ 的单位是 μm$^{-1}$，与 $f_{\text{limit}}$ 一致。

Matsushima 公式中的关键比值 $\lambda \cdot \Delta z / L$ 中，$\lambda$ 和 $\Delta z$ 都统一到 μm 后，$L$ 也是 μm，比值为无量纲量。

---

## 6. 参数对截止频率的影响

固定 $\lambda = 532\text{nm}, \Delta x = 10\mu\text{m}, N=256$：

| $\Delta z$ (mm) | $L$ (mm) | $\lambda\Delta z / L$ | $\sqrt{1+(\dots)^2}$ | $f_{\text{limit}}$ (mm$^{-1}$) | 截止周期 (mm) |
|----------------|---------|---------------------|---------------------|------------------------------|--------------|
| 1 | 2.56 | 0.208 | 1.02 | 49.0 | 0.020 |
| 10 | 2.56 | 2.08 | 2.31 | 21.6 | 0.046 |
| 100 | 2.56 | 20.8 | 20.8 | 2.40 | 0.417 |
| 500 | 2.56 | 104 | 104 | 0.48 | 2.08 |

传播距离增大时截止频率急剧下降。在 $\Delta z=500\text{mm}$ 时，能正确传播的最细节周期为 2.08mm——这远大于网格的奈奎斯特极限 0.02mm。也就是说，**带限在远场情形下远比简单的奈奎斯特条件更严格**。

---

## 7. 参考资料

1. Matsushima, K., & Shimobaba, T. (2009). "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." *Optics Express*, 17(22), 19662-19673.
   - 核心论文，提出截止频率公式，并证明该滤波器能严格防止混叠

2. Voelz, D. G. (2011). *Computational Fourier Optics: A MATLAB Tutorial*. SPIE Press.
   - 角谱法的教科书级介绍，包含离散实现细节

3. Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.). Roberts & Company.
   - 傅里叶光学的经典教材，角谱法的理论基础

---

*本文档对应代码 `js/physics/angspec.js`，截图位于 `screenshots/`。*
