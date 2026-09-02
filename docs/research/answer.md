# 多层超表面系统中非相干光的建模与优化

一般做法不是给“非相干光场”寻找一个确定复振幅，而是传播它的二阶统计量——互强度/交叉谱密度。对多层超表面优化，最实用的实现则是把互强度写成若干相干模态，继续使用现有的角谱传播和自动微分。

你的方案 1 是精确模型；方案 2 是它的随机无偏近似。困难主要在计算规模，而不是物理模型本身。

## 1. 统一的数学形式

把离散输入复光场写成列向量 $\mathbf u$。第 $\ell$ 层超表面为

$$
\mathbf M_\ell(\theta_\ell)
=
\operatorname{diag}
\left(e^{i\theta_\ell(\mathbf r)}\right),
$$

层间角谱传播算子为

$$
\mathbf P_d
=
\mathbf F^{-1}
\operatorname{diag}\!\left[
H_d(k_x,k_y)
\right]\mathbf F,
$$

其中

$$
H_d(k_x,k_y)
=
\exp\left(
id\sqrt{k^2-k_x^2-k_y^2}
\right).
$$

整个级联系统是线性复振幅算子

$$
\mathbf U_{\boldsymbol\theta}
=
\mathbf P_{L}
\mathbf M_L
\cdots
\mathbf P_2
\mathbf M_2
\mathbf P_1
\mathbf M_1.
$$

相干光下：

$$
\mathbf u_{\rm out}
=
\mathbf U_{\boldsymbol\theta}\mathbf u_{\rm in},
\qquad
\mathbf I_{\rm out}
=
|\mathbf u_{\rm out}|^2.
$$

非相干或部分相干光则用输入互强度矩阵

$$
\mathbf J_{\rm in}
=
\mathbb E[
\mathbf u_{\rm in}\mathbf u_{\rm in}^{\dagger}
].
$$

其传播规律非常简洁：

$$
\boxed{
\mathbf J_{\rm out}
=
\mathbf U_{\boldsymbol\theta}
\mathbf J_{\rm in}
\mathbf U_{\boldsymbol\theta}^{\dagger}
}
$$

探测器测得的是对角线：

$$
\boxed{
\mathbf I_{\rm out}
=
\operatorname{diag}(\mathbf J_{\rm out})
}.
$$

这就是非相干光通过任意多层线性光学系统的核心模型。

不要真的存储 $\mathbf J$：二维网格有 $N$ 个像素时，它有 $N^2$ 个复数，传播也会变得极其昂贵。下面几种分解就是为了绕开它。

## 2. 完全空间非相干光：点源求和是精确答案

若源面各像素严格互不相干，输入互强度为

$$
J_{\rm in}(\mathbf r_1,\mathbf r_2)
=
S(\mathbf r_1)\delta(\mathbf r_1-\mathbf r_2),
$$

离散后就是

$$
\mathbf J_{\rm in}
=
\operatorname{diag}(\mathbf s),
$$

其中 $\mathbf s$ 是输入强度。

于是

$$
I_{{\rm out},q}
=
\sum_p
s_p
\left|
U_{\boldsymbol\theta,qp}
\right|^2.
$$

也就是

$$
\boxed{
\mathbf I_{\rm out}
=
|\mathbf U_{\boldsymbol\theta}|^{\circ 2}\mathbf s
}
$$

这里 $|\cdot|^{\circ2}$ 表示逐元素模平方。

因此你的方案 1 完全正确：

$$
I_{\rm out}(\mathbf r)
=
\int
S(\boldsymbol\rho)
\left|
h_{\boldsymbol\theta}
(\mathbf r;\boldsymbol\rho)
\right|^2
d\boldsymbol\rho.
$$

注意多层任意相位板通常是空间变系统，因此 $h(\mathbf r;\boldsymbol\rho)$ 不一定只依赖 $\mathbf r-\boldsymbol\rho$，一般不能简化成一次普通卷积。

优点：

- 物理上精确；
- 完全可微；
- 不需要随机平均；
- 适合作为验证基准。

缺点是有 $N_{\rm source}$ 个点源时，每次优化迭代大约需要 $N_{\rm source}$ 次级联传播。

实际实现中可以将点源维度作为 batch，分块传播，而不是一次处理一个点。

## 3. 最适合大规模训练的方案：随机相位迹估计

令

$$
u_m(\mathbf r)
=
\sqrt{S(\mathbf r)}
\exp[i\phi_m(\mathbf r)],
$$

且不同像素的相位独立满足

$$
\phi_m(\mathbf r)\sim \mathcal U(0,2\pi).
$$

那么

$$
\mathbb E[
\mathbf u_m\mathbf u_m^\dagger
]
=
\operatorname{diag}(\mathbf s).
$$

因此

$$
\boxed{
\mathbf I_{\rm out}
=
\mathbb E_{\phi}
\left[
\left|
\mathbf U_{\boldsymbol\theta}
\left(
\sqrt{\mathbf s}\odot e^{i\boldsymbol\phi}
\right)
\right|^2
\right]
}
$$

可以用 $M$ 次随机实验估计：

$$
\widehat{\mathbf I}_{\rm out}
=
\frac{1}{M}
\sum_{m=1}^{M}
\left|
\mathbf U_{\boldsymbol\theta}
\mathbf u_m
\right|^2.
$$

这不是经验性的“模拟杂乱随机相位”，而是对

$$
\operatorname{diag}
\left(
\mathbf U
\operatorname{diag}(\mathbf s)
\mathbf U^\dagger
\right)
$$

所做的随机迹估计。对平均输出强度而言它是无偏的。

每个随机样本仍然走你原来的相干传播链：

```text
sqrt(input_intensity) * exp(i * random_phase)
→ metasurface
→ angular-spectrum propagation
→ metasurface
→ ...
→ abs(field)**2
→ average over random realizations
→ loss
→ automatic differentiation
```

伪代码大致是：

```python
phase = 2 * pi * rand(B, H, W)
u = sqrt(source_intensity)[None] * exp(1j * phase)

for theta, distance in layers:
    u = u * exp(1j * theta)
    u = angular_spectrum(u, distance)

I = mean(abs(u)**2, dim=0)
loss = criterion(I, target)
loss.backward()
```

推荐训练策略：

- 每次迭代重新生成随机相位；
- 初期用较小 $M$，如 $4\sim16$；
- 后期增加到 $32\sim128$；
- 最终用点源精确求和或很大的 $M$ 验证；
- 多个随机场必须先平均强度，再计算代表时间平均探测结果的 loss。

如果直接计算

$$
\frac1M\sum_m \mathcal L(I_m)
$$

它对应的是“瞬时散斑损失的平均”，通常不等于

$$
\mathcal L\left(\frac1M\sum_m I_m\right).
$$

真正的慢探测器通常对应后者。

### 如何进一步降低随机方差

比完全独立随机相位更好的选择包括：

- Hadamard 正交编码；
- 随机正交相位向量；
- 分层抽样；
- quasi-Monte Carlo；
- 每隔若干步使用较大 batch 校正梯度。

若使用完整 $N$ 阶 Hadamard 基并对全部编码求平均，结果可以恢复精确的非相干点源求和；取其中一小部分则是低方差近似。

## 4. 部分相干光：相干模态分解通常最好

一般部分相干源满足

$$
\mathbf J_{\rm in}
=
\mathbf \Phi
\mathbf \Lambda
\mathbf \Phi^\dagger
=
\sum_{n}
\lambda_n
\boldsymbol\phi_n
\boldsymbol\phi_n^\dagger.
$$

每个 $\boldsymbol\phi_n$ 是一个相干模式。输出强度为

$$
\boxed{
\mathbf I_{\rm out}
=
\sum_n
\lambda_n
\left|
\mathbf U_{\boldsymbol\theta}
\boldsymbol\phi_n
\right|^2
}.
$$

所以只需要分别传播若干相干模式，然后加权求强度和。Wolf 的相干模态理论正是这一表示的基础：[Wolf 1982](https://doi.org/10.1364/JOSA.72.000343)、[Wolf 1986](https://doi.org/10.1364/JOSAA.3.001920)。

若特征值快速衰减，只保留前 $R$ 个模式：

$$
\mathbf I_{\rm out}
\approx
\sum_{n=1}^{R}
\lambda_n
\left|
\mathbf U_{\boldsymbol\theta}
\boldsymbol\phi_n
\right|^2.
$$

误差可由舍弃的能量衡量：

$$
\epsilon_{\rm mode}
=
\frac{\sum_{n>R}\lambda_n}
{\sum_n\lambda_n}.
$$

这通常是部分相干光最好的建模方式，并且天然兼容自动微分。Gaussian–Schell 模型等常见光源还具有解析或近解析模态；有效模式数与光源尺寸和相干长度之比密切相关：[Starikov 与 Wolf](https://doi.org/10.1364/JOSA.72.000923)。

不过，对严格完全非相干的 $N$ 像素源，

$$
\mathbf J_{\rm in}=\operatorname{diag}(\mathbf s)
$$

通常具有接近 $N$ 的秩，模态分解不能天然降低复杂度。这时随机迹估计更实用。

## 5. 一个容易忽略但非常重要的限制

如果第一层相位超表面恰好位于严格空间非相干的源平面，那么这层纯相位调制完全不起作用。

因为

$$
\mathbf M_1
\mathbf J_{\rm in}
\mathbf M_1^\dagger
=
\operatorname{diag}(e^{i\theta_1})
\operatorname{diag}(\mathbf s)
\operatorname{diag}(e^{-i\theta_1})
=
\operatorname{diag}(\mathbf s).
$$

连续形式也是

$$
e^{i\theta(\mathbf r_1)}
S(\mathbf r_1)\delta(\mathbf r_1-\mathbf r_2)
e^{-i\theta(\mathbf r_2)}
=
S(\mathbf r_1)\delta(\mathbf r_1-\mathbf r_2).
$$

所以

$$
\frac{\partial I_{\rm out}}{\partial\theta_1}=0.
$$

这不是优化器问题，而是物理上的不可辨识性。

第一层相位要产生作用，至少需要以下条件之一：

- 入射场具有有限空间相干长度；
- 源到第一层之间先经过一段传播，使第一层面出现非零互相干；
- 第一层具有振幅调制；
- 超表面响应具有空间耦合，而不是局域薄相位近似；
- 对不同入射角、偏振或波长有不同响应。

后续层仍然可以有作用，因为传播后即使初始场是空间非相干的，互强度一般也不再保持对角形式：

$$
\mathbf J_1
=
\mathbf P
\operatorname{diag}(\mathbf s)
\mathbf P^\dagger.
$$

## 6. 我建议的最终方案

### 完全非相干、单色、标量模型

训练时采用：

$$
\widehat I(\theta)
=
\frac1M
\sum_{m=1}^{M}
\left|
U_\theta
\left(
\sqrt S\odot e^{i\phi_m}
\right)
\right|^2,
$$

使用随机相位或正交编码，直接自动微分。

验证时采用：

$$
I_q
=
\sum_p S_p|U_{qp}|^2,
$$

通过点源 batch 分块精确计算。

这是精度、显存和训练时间之间最合理的组合，也与已有的非相干衍射网络建模一致。相关工作明确研究了任意空间相干性下衍射网络的高效模拟与训练：[Filipovich 等，Optics Express 2024](https://arxiv.org/abs/2310.03679)。空间非相干系统也可以明确写成 $H=|h|^2$ 的线性强度变换：[Universal Linear Intensity Transformations](https://arxiv.org/abs/2303.13037)。

### 部分相干光

若已知或能标定 $J_{\rm in}$，优先做截断相干模态分解：

$$
J_{\rm in}\approx
\sum_{n=1}^{R}\lambda_n\phi_n\phi_n^\dagger.
$$

将 $R$ 个模式作为 batch 传播，输出强度加权求和。这通常比随机实验稳定。

### 宽谱非相干光

不同波长之间通常也应按强度求和：

$$
I_{\rm detector}
=
\sum_j w_j
\sum_n\lambda_{j,n}
\left|
U_{\theta,\lambda_j}
\phi_{j,n}
\right|^2.
$$

需要注意：

- 每个波长的角谱传递函数不同；
- 超表面相位响应一般随波长变化，不能简单把同一个 $\theta$ 用在所有波长；
- 若偏振不相干，也对两个偏振通道分别计算后加强度。

## 7. 关于你式子里的角谱范围

传播模态的区域应写成

$$
k_x^2+k_y^2\le k^2,
$$

而不是分别限制

$$
0<k_x,k_y<k.
$$

FFT 频率同时包含正负 $k_x,k_y$。传播区是频率平面中的圆盘。圆盘之外：

$$
k_z=i\sqrt{k_x^2+k_y^2-k^2},
$$

对应倏逝波，传播因子为指数衰减；宏观层间距下通常直接截断即可。这个频谱不对称问题与非相干性没有直接关系，非相干性应通过 $J$ 或统计系综处理，而不是试图在单次角谱积分里消去随机相位。

简而言之：以互强度作为理论起点，以“相干模态传播”作为统一计算框架；严格非相干时训练使用随机相位/正交编码的随机迹估计，验证使用点源强度精确求和。这套模型完全支持 $\theta_1,\theta_2,\ldots$ 的自动微分，同时还会正确揭示某些相位层在严格非相干条件下实际上没有设计自由度。
