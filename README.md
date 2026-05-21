# AFT-DINO

Official implementation of **AFT-DINO: Improved Self-Supervised Pretraining and Adaptation for Fabric Defect Detection**.

AFT-DINO adapts DINO-style self-supervised pretraining to lightweight YOLO11n backbones for fabric defect detection. The repository is organized as a paper algorithm package with three explicit method modules: LSA-Mix, HAFD, and TBA Loss.

## Core modules

- **LSA-Mix**: Local Statistics-Aligned Mixing for local texture perturbation in DINO local views.
- **HAFD**: Hierarchical Adaptive Feature Fusion Distillation with layer adapters and GateBlock-based adaptive fusion.
- **TBA Loss**: Optimal Transport Balanced Assignment loss for balanced teacher target generation.

## Repository structure

```text
AFT-DINO/
├── aft_dino/
│   ├── models/
│   │   ├── aft_dino.py
│   │   └── pool_head.py
│   ├── hafd/
│   │   ├── adapter.py
│   │   ├── gate_block.py
│   │   └── hafd.py
│   ├── transforms/
│   │   └── lsa_mix.py
│   ├── losses/
│   │   └── tba_loss.py
│   └── engine/
│       └── trainer.py
├── configs/
│   └── config.py
├── scripts/
│   └── pretrain.py
├── models/
├── TILDA/
├── requirements.txt
└── README.md
```

## Module mapping

| Method component | Implementation |
|---|---|
| AFT-DINO student-teacher framework | `aft_dino/models/aft_dino.py` |
| HAFD layer adapter | `aft_dino/hafd/adapter.py` |
| HAFD adaptive fusion block | `aft_dino/hafd/gate_block.py` |
| HAFD wrapper | `aft_dino/hafd/hafd.py` |
| Pooling head | `aft_dino/models/pool_head.py` |
| Self-supervised training loop | `aft_dino/engine/trainer.py` |
| Pretraining entry point | `scripts/pretrain.py` |
| LSA-Mix transform | `aft_dino/transforms/lsa_mix.py` |
| TBA Loss | `aft_dino/losses/tba_loss.py` |

The vendored `lightly/` folder is intentionally not included. This package depends on the official `lightly` package through `requirements.txt` and keeps only AFT-DINO-specific method code in `aft_dino/`.

## Installation

```bash
pip install -r requirements.txt
# Optional: install the local package in editable mode
pip install -e .
```

Place the YOLO11n weight file here:

```text
models/yolo11n.pt
```

Prepare the unlabeled pretraining images in the following layout:

```text
TILDA/images/train/
TILDA/images/val/
```

## Pretraining

```bash
python scripts/pretrain.py
```

The script inserts the project root into `sys.path`, so it can be run directly from the repository root without requiring editable installation.

The pretrained model will be saved to:

```text
models/aft_dino_yolo11n_pretrained.pt
```

## Python imports

```python
from aft_dino.models import AFTDINO, PoolHead
from aft_dino.hafd import HAFD, BackboneLayerAdapter, GateBlock
from aft_dino.transforms import LSAMixDINOTransform
from aft_dino.losses import TBADINOLoss
```

Backward-compatible aliases are kept for minimal disruption:

```python
from aft_dino.models import DINO
from aft_dino.transforms import DINOTransform
from aft_dino.losses import DINOLoss
```

## Notes

- The downstream YOLO detection architecture is not changed by AFT-DINO.
- The main algorithmic changes are applied during self-supervised pretraining.
- Dataset paths and hyperparameters are kept close to the original experimental setting to minimize changes to the training logic.
