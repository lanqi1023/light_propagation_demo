# Incoherent metasurface propagation

本仓库用于研究 MicroLED 非相干光经过多层超表面后的可微传播，以及该系统作为神经网络非负线性层时的可实现性。

## 目录

- [`incoherent_metasurface_modes/`](incoherent_metasurface_modes/)：当前代码项目，包含 PyTorch 模型、测试、演示与可视化结果。
- [`incoherent_metasurface_modes/MATHEMATICAL_MODEL.md`](incoherent_metasurface_modes/MATHEMATICAL_MODEL.md)：从朗伯 MicroLED、准直 metalens、角谱传播到平方律探测的数学过程。
- [`docs/research/discussion.md`](docs/research/discussion.md)：完整讨论记录。
- [`docs/research/main_idea.md`](docs/research/main_idea.md)：已确认结论、风险和不可行点。
- [`docs/research/answer.md`](docs/research/answer.md)：早期关于非相干光建模方案的系统回答。

## 快速验证

```bash
cd incoherent_metasurface_modes
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python examples/simple_demo.py
PYTHONPATH=src .venv/bin/python examples/visualize_demo.py
```

依赖、参数和其他运行方式见 [`incoherent_metasurface_modes/README.md`](incoherent_metasurface_modes/README.md)。
