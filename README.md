# PyTorch Template

- Package manager: pixi
- Deep Learning Framework: Lightning
- Logger: W&B

# Installation

```bash
pixi install
```

## Verification

```bash
pixi run test_pytorch
```

# Train model

```bash
pixi run python train.py

# or you can use task defined at pixi.toml
pixi run train
```