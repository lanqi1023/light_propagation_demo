# 朗伯 MicroLED 阵列经多层超表面传播的数学模型

本文对应 `metasurface_modes.torch_api` 中的 PyTorch 实现。目标是建立如下可微系统：

$$
\text{非负 LED 功率向量 }\mathbf x
\longrightarrow
\text{朗伯源混合模式}
\longrightarrow
\text{准直 metalens}
\longrightarrow
\text{多层可训练相位面}
\longrightarrow
\text{探测器功率 }\mathbf y.
$$

模型采用准单色、标量、线性光学近似。LED 之间以及单颗 LED 内选取的发光子点之间均按统计不相干处理；每个相干模式仍使用波动光学传播。

---

## 1. 坐标、采样与功率归一化

所有代码使用 SI 单位。横向计算网格为

$$
x_n=\left(n-\frac{N_x-1}{2}\right)\Delta x,
\qquad
y_m=\left(m-\frac{N_y-1}{2}\right)\Delta y.
$$

连续场 $E(x,y)$ 离散为功率归一化场

$$
u[m,n]=E(x_n,y_m)\sqrt{\Delta x\Delta y}.
$$

因此离散场的平方范数直接表示该平面上的功率：

$$
P=\sum_{m,n}|u[m,n]|^2.
$$

传播使用正交归一化 FFT，即 PyTorch 的 `norm="ortho"`。当所有离散空间频率均处在传播圆盘内、没有孔径截断且不考虑吸收时，角谱传播保持上述离散功率。

---

## 2. MicroLED 阵列与输入变量

设阵列有 $N_{\rm LED}$ 颗 LED，第 $i$ 颗的输入功率为

$$
x_i\ge0.
$$

神经网络输入是

$$
\mathbf x=(x_1,\ldots,x_{N_{\rm LED}})^\mathsf T.
$$

当前讨论中的完整系统为

$$
N_{\rm LED}=4096\times12=49\,152,
$$

但示例程序使用小阵列，以便在普通机器上验证物理和梯度。

LED 中心坐标为 $\boldsymbol\rho_i=(x_i^{\rm LED},y_i^{\rm LED})$。注意这里上标 `LED` 用于区分位置与输入功率 $x_i$。

---

## 3. 朗伯发光模型

### 3.1 单个点发射体

归一化为总辐射功率 1 的理想朗伯点源，其单位立体角辐射强度为

$$
\frac{\mathrm dP}{\mathrm d\Omega}
=\frac{1}{\pi}\cos\theta,
\qquad 0\le\theta\le\frac{\pi}{2}.
$$

源平面位于第一超表面前方距离 $f$，两平面互相平行。源点 $s$ 到第一层位置 $\mathbf r=(x,y)$ 的距离为

$$
R_s(\mathbf r)
=\sqrt{f^2+(x-x_s)^2+(y-y_s)^2},
$$

且

$$
\cos\theta_s(\mathbf r)=\frac{f}{R_s(\mathbf r)}.
$$

朗伯辐射强度给出一个 $\cos\theta$，接收平面微元对应的立体角再给出一个投影因子 $\cos\theta/R^2$。因此第一层上的辐照度为

$$
I_s^-(\mathbf r)
=\frac{1}{\pi}
\frac{\cos^2\theta_s(\mathbf r)}{R_s^2(\mathbf r)}.
$$

选择与该辐照度一致的球面波模式

$$
\psi_s^-(\mathbf r)
=P(\mathbf r)
\frac{1}{\sqrt\pi}
\frac{\cos\theta_s(\mathbf r)}{R_s(\mathbf r)}
\exp\!\left[ikR_s(\mathbf r)\right],
\qquad
k=\frac{2\pi}{\lambda},
$$

其中 $P(\mathbf r)$ 是第一层有限孔径。

离散代码把 $\sqrt{\Delta x\Delta y}$ 乘进模式振幅，所以

$$
\|\boldsymbol\psi_s^-\|_2^2
\approx
\int_{\rm aperture}I_s^-(\mathbf r)\,\mathrm d^2\mathbf r
$$

就是该源点被第一层收集的功率比例。代码没有把每个模式强行归一化为 1，因此有限 NA 的收集损失会保留下来。

对半径 $a$ 的圆孔和轴上点源，上式的连续积分为

$$
\eta_{\rm coll}
=\frac{a^2}{a^2+f^2}
=\sin^2\theta_{\max}
=\mathrm{NA}^2
$$

（空气中）。这与讨论中的朗伯源收集效率估计一致。

当前 `simple_demo.py` 和 `visualize_demo.py` 使用毫米量级矩形有效孔径：

$$
P(x,y)=
\begin{cases}
1,& |x|\le W/2\ \text{且}\ |y|\le H/2,\\
0,& \text{其他位置},
\end{cases}
$$

其中 $W=1.2\,\mathrm{mm}$、$H=1.0\,\mathrm{mm}$。配套参数为 $f=20\,\mathrm{mm}$、$216\times192$ 网格和 $6\,\mu\mathrm m$ 采样间隔。增大孔径时同步扩大计算窗口和调整焦距/采样，是为了让整个矩形孔径落入网格并满足最大传播角的 Nyquist 采样。圆孔公式仍保留用于解释 NA 和收集效率；代码同时支持圆形、矩形或用户给定的任意非负孔径 mask。

### 3.2 有限发光面积

第 $i$ 颗 LED 的发光面可离散为 $S$ 个子点，模式写为 $\psi_{i,s}$。若假设发光面均匀，每个子点承担该 LED 功率的 $1/S$，代码把 $1/\sqrt S$ 乘入模式振幅。

该 LED 的互强度为

$$
\mathbf J_i^-
=\frac1S\sum_{s=1}^{S}
\boldsymbol\psi_{i,s}^-
\boldsymbol\psi_{i,s}^{-\dagger}.
$$

实现中经 $1/\sqrt S$ 缩放后的模式仍记作 $\psi_{i,s}$，于是无需在后续公式重复写 $1/S$。

`emitter_samples_y=emitter_samples_x=1` 是点 LED 近似。增大采样数可以表达有限发光面积造成的部分空间相干性下降，但计算量与模式数成正比。

---

## 4. 非相干输入的互强度

不同 LED 以及不同发光子点之间统计独立。总输入互强度为

$$
\boxed{
\mathbf J_{\rm in}^-(\mathbf x)
=\sum_{i=1}^{N_{\rm LED}}
x_i\sum_{s=1}^{S}
\boldsymbol\psi_{i,s}^-
\boldsymbol\psi_{i,s}^{-\dagger}
}.
$$

这一步说明：

- 不把不同 LED 的确定复振幅直接相加；
- LED 功率 $x_i$ 对互强度是线性的；
- 每个 $\psi_{i,s}$ 仍可用普通复数波动传播；
- 不需要显式构造尺寸为 $N_{\rm grid}^2$ 的互强度矩阵。

---

## 5. 第一层准直 metalens

LED 平面到第一层的距离设为第一层焦距 $f$。轴上点源球面相位为 $k\sqrt{f^2+x^2+y^2}$。第一层使用双曲准直相位

$$
\phi_{\rm lens}(x,y)
=-k\left(\sqrt{f^2+x^2+y^2}-f\right).
$$

相位透射函数为

$$
M_0(x,y)
=P(x,y)\exp[i\phi_{\rm lens}(x,y)].
$$

对轴上源，$M_0$ 抵消球面波相位并留下常相位；对离轴 LED，它在傍轴近似下产生不同倾角的准直模式。

代码还支持讨论中确定的统一形式

$$
\phi_0=\phi_{\rm lens}+\Delta\phi_0.
$$

默认 `trainable_collimator_residual=False`，即固定准直层；设为 `True` 时，$\Delta\phi_0$ 从零初始化并可训练。

第一层后的单模式为

$$
\boldsymbol\psi_{i,s}^{+}
=\mathbf M_0\boldsymbol\psi_{i,s}^{-}.
$$

---

## 6. 层间角谱传播

对传播距离 $d_\ell$，二维 Fourier 域传递函数为

$$
H_{d_\ell}(k_x,k_y)
=\exp\!\left[
i d_\ell\sqrt{k^2-k_x^2-k_y^2}
\right],
$$

其传播波支持为

$$
k_x^2+k_y^2\le k^2.
$$

代码默认丢弃该圆盘外的倏逝分量。离散传播算子为

$$
\boxed{
\mathcal P_{d_\ell}[u]
=\mathcal F^{-1}
\left\{H_{d_\ell}\,\mathcal F[u]\right\}
}.
$$

FFT 网格有周期边界。物理场可能传播到计算窗口外时，需要在调用模型前扩大网格或 padding；代码不会自动判断 wrap-around。

---

## 7. 后续可训练超表面

第 $\ell$ 个可训练层是局域纯相位面

$$
M_\ell(x,y)
=P_\ell(x,y)\exp[i\theta_\ell(x,y)],
\qquad \ell=1,\ldots,L.
$$

$\theta_\ell$ 在 PyTorch 中直接存为 `nn.Parameter`。训练时不必显式执行 $\bmod 2\pi$，因为

$$
e^{i(\theta+2\pi n)}=e^{i\theta}.
$$

制造导出时再做 $2\pi$ 映射、相位量化和 meta-atom 几何映射。

对任一输入相干模式，完整传播为

$$
\mathbf U_{\boldsymbol\theta}
=\mathbf P_{d_L}\mathbf M_L
\cdots
\mathbf P_{d_1}\mathbf M_1
\mathbf P_{d_0}\mathbf M_0.
$$

代码中的 `propagation_distances_m` 长度必须为 $L+1$：准直层到第一训练层一段，各训练层之间若干段，最后一层到探测器一段。

---

## 8. 探测器

第 $j$ 个探测器用非负空间权重 $D_j[m,n]$ 表示。对一个相干输出场 $u$，读数为

$$
y_j[u]
=\sum_{m,n}D_j[m,n]|u[m,n]|^2.
$$

`tiled_detector_masks` 把整个输出网格无重叠地划分为矩形区域；也可以自行提供任意非负 mask 来描述不规则探测区域、像素响应或加权积分。

---

## 9. 精确非相干前向过程

第 $(i,s)$ 个模式经过整个系统后为

$$
\mathbf v_{i,s}
=\mathbf U_{\boldsymbol\theta}\boldsymbol\psi_{i,s}^{-}.
$$

矩阵元素为

$$
\boxed{
A_{ji}(\boldsymbol\theta)
=\sum_s
\left\|\mathbf D_j^{1/2}\mathbf v_{i,s}\right\|_2^2
\ge0
}.
$$

所有不相干模式按强度相加：

$$
\boxed{
y_j=\sum_iA_{ji}(\boldsymbol\theta)x_i,
\qquad
\mathbf y=\mathbf A_{\boldsymbol\theta}\mathbf x
}.
$$

代码路径是

```python
prediction = model(powers, method="exact")
```

它按 mode chunk 生成朗伯球面波、传播并累加强度，不保存全部模式。该方法没有随机误差，适合小系统、抽样验证和最终逐列审计；其时间仍与总源模式数成正比。

`model.intensity_matrix()` 可显式构造 $\mathbf A$，但仅应在小系统或离线分块诊断中使用。

---

## 10. 随机相位无偏估计

为避免每个训练样本逐模式传播，对每次随机实现 $q$ 生成

$$
c_{i,s}^{(q)}
=\sqrt{x_i}\exp(i\xi_{i,s}^{(q)}),
\qquad
\xi_{i,s}^{(q)}\sim\mathcal U[0,2\pi),
$$

并构造一个相干叠加场

$$
\mathbf u_{\rm in}^{(q)}
=\sum_{i,s}c_{i,s}^{(q)}\boldsymbol\psi_{i,s}^{-}.
$$

独立随机相位满足

$$
\mathbb E\left[
c_{i,s}^{(q)}c_{k,t}^{(q)*}
\right]
=x_i\delta_{ik}\delta_{st}.
$$

所以

$$
\mathbb E\left[
\mathbf u_{\rm in}^{(q)}
\mathbf u_{\rm in}^{(q)\dagger}
\right]
=\mathbf J_{\rm in}.
$$

传播 $K$ 个实现后，对探测强度取平均：

$$
\boxed{
\widehat y_j
=\frac1K\sum_{q=1}^{K}
\left\|
\mathbf D_j^{1/2}
\mathbf U_{\boldsymbol\theta}
\mathbf u_{\rm in}^{(q)}
\right\|_2^2
}.
$$

其期望严格等于精确非相干结果：

$$
\mathbb E[\widehat{\mathbf y}]
=\mathbf A_{\boldsymbol\theta}\mathbf x.
$$

代码路径是

```python
prediction = model(
    powers,
    method="stochastic",
    n_realizations=8,
    realization_chunk_size=8,
)
```

训练时必须先在 $K$ 个实现上平均探测强度，再对平均结果计算对应物理任务的非线性损失。增加 $K$ 可降低方差，但增加显存和计算量。

---

## 11. 自动微分

对任务损失

$$
\mathcal L
=\mathcal L(\widehat{\mathbf y},\mathbf y_{\rm target}),
$$

PyTorch 沿以下计算图求梯度：

$$
\theta_\ell
\longrightarrow e^{i\theta_\ell}
\longrightarrow \text{复场乘法}
\longrightarrow \text{FFT/IFFT}
\longrightarrow |u|^2
\longrightarrow \widehat{\mathbf y}
\longrightarrow \mathcal L.
$$

因此可以直接使用

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

默认仅后续 phase surfaces 可训练。若启用第一层残差，相同计算图也会更新 $\Delta\phi_0$。

---

## 12. 代码组成

- `torch_grid.py`：物理网格、坐标与空间频率；
- `torch_source.py`：有限面积朗伯 LED 阵列、孔径、随机相干实现和收集效率；
- `torch_optics.py`：角谱传播、相位面、准直 metalens、区域探测器；
- `torch_system.py`：多层系统、精确/随机非相干前向和强度矩阵；
- `torch_api.py`：PyTorch 公共导入入口；
- `examples/train_demo.py`：两层可训练相位面的最小训练示例；
- `tests/test_torch_propagation.py`：功率守恒、准直相位、线性矩阵、随机估计和梯度测试。

---

## 13. 安装与运行

在独立项目目录执行：

```bash
cd incoherent_metasurface_modes
python -m pip install -e '.[torch,test]'
pytest
PYTHONPATH=src python examples/simple_demo.py
PYTHONPATH=src python examples/visualize_demo.py
PYTHONPATH=src python examples/train_demo.py
```

最小构造方式为：

```python
import torch

from metasurface_modes.torch_api import (
    AreaDetector,
    CartesianGrid,
    IncoherentMetasurfaceModel,
    LambertianLEDArray,
    LambertianLEDArrayConfig,
    MultilayerMetasurface,
    rectangular_aperture,
    tiled_detector_masks,
)

wavelength = 532e-9
focal_length = 20e-3
grid = CartesianGrid(192, 216, 6e-6, 6e-6)
aperture = rectangular_aperture(
    grid,
    height_m=1.0e-3,
    width_m=1.2e-3,
)

source = LambertianLEDArray(
    LambertianLEDArrayConfig(
        layout_y=4,
        layout_x=4,
        pitch_y_m=30e-6,
        pitch_x_m=30e-6,
        source_to_metasurface_m=focal_length,
        emitter_height_m=4e-6,
        emitter_width_m=4e-6,
    ),
    aperture=aperture,
)

optical = MultilayerMetasurface(
    grid,
    wavelength,
    focal_length,
    n_trainable_layers=2,
    propagation_distances_m=(5e-3, 5e-3, 5e-3),
    aperture=aperture,
)

detector = AreaDetector(tiled_detector_masks(grid, 4, 4))
model = IncoherentMetasurfaceModel(source, optical, detector)

powers = torch.rand(8, 16)
output = model(powers, method="stochastic", n_realizations=8)
```

---

## 14. 适用范围和下一步扩展

当前实现是可验证的基准模型，不是 $49\,152$ 输入生产规模的最终高性能内核。主要边界如下：

1. **准单色标量模型**：未包含矢量偏振、meta-atom 非局域耦合和全波散射。
2. **理想相位层**：后续层默认单位振幅；真实器件应替换为随波长、角度和偏振变化的复透射 surrogate。
3. **朗伯近似**：真实 MicroLED 角分布、封装微透镜和反射结构应由测量数据替换。
4. **空间非相干发光面**：代码将 LED 子点视为互不相干；若单 LED 有已知部分相干性，应输入相干模态及权重。
5. **直接模式生成**：当前正确性优先的实现逐块生成球面波，复杂度约为 $O(N_{\rm modes}N_{\rm grid})$。完整 $49\,152$ 阵列训练应利用规则阵列的 Fresnel/Fourier 结构、正交编码、低秩模态或专用 CUDA 算子加速。
6. **固定波长**：宽谱 LED 应对多个波长分别传播后按谱功率求和。
7. **FFT 周期边界**：需要按传播距离、孔径和最大角度选择足够网格与 padding。
8. **输出维度**：当前 tiled detector 是示例。实际 $8192$ 个双轨输出需要明确探测区域、隐藏细模式和垃圾端口。

建议先在小阵列上用 `exact` 与 `stochastic` 交叉验证，再增加网格、LED 数、发光子点数和层数；不要一开始直接把 raw meta-atom 数当作传播网格大小。
