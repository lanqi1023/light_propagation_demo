# Incoherent metasurface propagation

本仓库用于研究 MicroLED 非相干光经过多层超表面后的可微传播，以及该系统作为神经网络非负线性层时的可实现性。

## 目录

- [`incoherent_metasurface_modes/`](incoherent_metasurface_modes/)：当前代码项目，包含 PyTorch 模型、测试、演示与可视化结果。
- [`incoherent_metasurface_modes/MATHEMATICAL_MODEL.md`](incoherent_metasurface_modes/MATHEMATICAL_MODEL.md)：从朗伯 MicroLED、准直 metalens、角谱传播到平方律探测的数学过程。
- [`docs/research/discussion.md`](docs/research/discussion.md)：完整讨论记录。
- [`docs/research/main_idea.md`](docs/research/main_idea.md)：已确认结论、风险和不可行点。
- [`docs/research/answer.md`](docs/research/answer.md)：早期关于非相干光建模方案的系统回答。

## 环境与运行

优先使用当前项目目录内的虚拟环境：

```bash
cd incoherent_metasurface_modes
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python examples/simple_demo.py
PYTHONPATH=src .venv/bin/python examples/visualize_demo.py
PYTHONPATH=src .venv/bin/python examples/train_demo.py
```

如果虚拟环境尚未创建：

```bash
cd incoherent_metasurface_modes
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[torch,test,viz]'
```

依赖、参数和其他运行方式见 [`incoherent_metasurface_modes/README.md`](incoherent_metasurface_modes/README.md)。

## 建模约定

- PyTorch 复数张量表示单个光源模式的相干光场。
- 相互非相干的 LED 或子发光点在探测器上叠加强度，不叠加复振幅。
- 第一层是具有矩形有效孔径的准直 metalens，并可选择添加可训练相位残差。
- 后续层是由角谱传播连接的可训练纯相位超表面。
- 探测器输出是各个互不重叠探测区域内的功率积分。
- 所有实现都应保持关于可训练相位参数的可微性。
- 精确的逐模式非相干求和是参考模型；随机相位方法只作为可选近似。

修改功率归一化、传播算子、光源采样或探测器积分前，应先阅读 [`MATHEMATICAL_MODEL.md`](incoherent_metasurface_modes/MATHEMATICAL_MODEL.md)。

## 仓库边界

本仓库目前只有 `incoherent_metasurface_modes/` 这一个活动代码项目。此前的 HTML/FastAPI 角谱传播演示已经移除，不应重新依赖原来的 `lightprop/`、`server.py`、`templates/`、`static/` 或根目录 `tests/`。
