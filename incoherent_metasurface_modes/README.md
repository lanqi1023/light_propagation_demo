# MicroLED 输入模式非正交性验证

这是一个独立于原光传播演示的验证项目，用来回答：

> 不同 MicroLED 虽然相互不相干，但它们在第一计算面产生的空间模式如果不正交，会不会限制可实现的非相干强度矩阵？

答案是会，而且可以定量判断。

## 物理模型

第 $i$ 个 LED 单独点亮时，在第一计算面产生归一化空间模式 $\psi_i$。定义

$$
G_{ik}=\langle\psi_i,\psi_k\rangle.
$$

共同的无损线性光学系统保持这个内积。若最终归一化输出强度列为 $a_i,a_k$，则必有

$$
\frac12\|a_i-a_k\|_1
\le\sqrt{1-|G_{ik}|^2}.
$$

所以：

- $|G_{ik}|=0$：这对输入没有额外的两两可区分性限制；
- $|G_{ik}|\approx1$：对应输出矩阵两列必须非常相似；
- $|G_{ik}|=1$：对应两列必然完全相同。

相互不相干只负责消除同时点亮时的干涉交叉项，并不会给光学系统附加一个可读取的“LED 来源标签”。

## 当前实现

- 理想透镜前焦面的位置—角度映射；
- 均匀矩形或圆形瞳面的解析模式重叠；
- 不构造 $49\,152^2$ Gram 矩阵的 $O(N_{\rm LED})$ 有效秩计算；
- 对任意数值仿真模式子集构造 Gram 矩阵；
- 对目标矩阵两列进行必要可行性检查；
- 用二维最优无损测量验证可区分度上界能够达到、不能突破。

## 快速运行

在本目录下执行：

```bash
PYTHONPATH=src python3 -m metasurface_modes.cli
```

默认使用 $256\times192=49\,152$ 个 LED，但所有物理尺寸都只是演示值。请替换为真实参数：

```bash
PYTHONPATH=src python3 -m metasurface_modes.cli \
  --layout-x 256 \
  --layout-y 192 \
  --pitch-x-um 10 \
  --pitch-y-um 10 \
  --wavelength-nm 532 \
  --focal-length-mm 20 \
  --pupil rectangle \
  --pupil-width-mm 1.064 \
  --pupil-height-mm 1.064
```

圆形孔径：

```bash
PYTHONPATH=src python3 -m metasurface_modes.cli \
  --pupil circle \
  --pupil-diameter-mm 2
```

机器可读输出：

```bash
PYTHONPATH=src python3 -m metasurface_modes.cli --json
```

运行测试：

```bash
pytest
```

## 需要替换的参数

在对实际系统下结论前，至少需要：

1. 49,152 颗 LED 的真实二维排布；
2. LED 中心间距；
3. 中心波长和谱宽；
4. 等效焦距；
5. 第一计算面上的有效孔径形状和尺寸；
6. 单颗 LED 的有限发光面积和角分布。

当前解析模型先把每颗 LED 视为单一相干空间模式。后续可把实测或全传播仿真的单 LED 场送入 `gram_matrix_from_modes`，验证非理想瞳面、像差和有限 LED 尺寸。

## PyTorch 多层非相干传播模型

本项目现已加入与上述模式诊断相互独立的可微波动模型，包括：

- 有限面积朗伯 MicroLED 阵列；
- 第一层固定准直 metalens，以及可选的可训练相位残差；
- 任意数量的后续可训练纯相位超表面；
- FFT 角谱传播；
- 区域平方律探测器；
- 逐模式精确非相干求和；
- 随机相位无偏估计；
- PyTorch 自动微分和示例训练程序。

数学推导、功率归一化和代码对应关系见 [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md)。

安装 PyTorch 可选依赖并运行：

```bash
python -m pip install -e '.[torch,test]'
pytest
PYTHONPATH=src python examples/simple_demo.py
PYTHONPATH=src python examples/visualize_demo.py
PYTHONPATH=src python examples/train_demo.py
```

建议先运行 `simple_demo.py`。它使用 $1.2\,\mathrm{mm}\times1.0\,\mathrm{mm}$ 的矩形有效孔径、$20\,\mathrm{mm}$ 焦距、$216\times192$ 传播网格、$2\times2$ LED 和 $2\times2$ 探测器，依次验证精确非相干传播等价于 $\mathbf A\mathbf x$、随机相位估计逼近精确结果，以及相位参数能够通过梯度下降拟合一个可实现的 teacher 系统。`train_demo.py` 则演示对一般非负目标的随机训练；一般目标不保证能由给定层数精确实现。

`visualize_demo.py` 会运行同一个可验证模型，并在 `artifacts/` 中输出光场、矩阵训练和精确/随机传播对比图。若按可选依赖安装，可使用 `python -m pip install -e '.[torch,test,viz]'` 一次安装绘图环境。

PyTorch API 单独放在 `metasurface_modes.torch_api` 中，因此只使用原有 NumPy/SciPy 模式诊断时不会强制导入 PyTorch。
