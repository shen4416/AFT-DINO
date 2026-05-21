import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

# Allow direct execution with: python scripts/pretrain.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    import torch
    from lightly.data import LightlyDataset
    from ultralytics import YOLO

    from aft_dino.engine import ssl_train
    from aft_dino.losses import DINOLoss
    from aft_dino.models import DINO, PoolHead
    from aft_dino.transforms import DINOTransform
    from configs.config import GLOBAL_CROP_SIZE, LOCAL_CROP_SIZE, LEARNING_RATE, BATCH_SIZE, EPOCHS

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # 1. Build the YOLO11n backbone and replace the final layer with PoolHead
    # -------------------------------------------------------------------------
    yolo = YOLO("models/yolo11n.pt")

    # Keep the first 12 YOLO layers as the self-supervised backbone
    yolo.model.model = yolo.model.model[:12]

    # Infer the feature dimension of the truncated backbone with a dummy input
    dummy = torch.rand(2, 3, GLOBAL_CROP_SIZE, GLOBAL_CROP_SIZE)

    # Infer the channel dimension before replacing the final layer
    backbone_without_last = yolo.model.model[:-1]
    out = backbone_without_last(dummy)

    # Replace the final layer with PoolHead to obtain compact feature maps
    yolo.model.model[-1] = PoolHead(
        yolo.model.model[-1].f,
        yolo.model.model[-1].i,
        out.shape[1],
    )

    # Compute the flattened feature dimension for the DINO projection head
    out = yolo.model(dummy)
    input_dim = out.flatten(start_dim=1).shape[1]

    # -------------------------------------------------------------------------
    # 2. Build AFT-DINO with student/teacher branches and HAFD
    # -------------------------------------------------------------------------
    backbone = yolo.model.requires_grad_()
    backbone.train()
    model = DINO(backbone, input_dim)
    model = model.to(device)

    # -------------------------------------------------------------------------
    # 3. Build LSA-Mix DINO transforms and datasets
    # -------------------------------------------------------------------------
    # YOLO-compatible normalization
    normalize = dict(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    transform = DINOTransform(
        global_crop_size=GLOBAL_CROP_SIZE,
        local_crop_size=LOCAL_CROP_SIZE,
        normalize=normalize,
    )

    # Load unlabeled images with LightlyDataset
    dataset_train = LightlyDataset(
        input_dir="TILDA/images/train",
        transform=transform,
    )
    dataset_val = LightlyDataset(
        input_dir="TILDA/images/val",
        transform=transform,
    )

    # Combine train and validation image folders for self-supervised pretraining
    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    # -------------------------------------------------------------------------
    # 4. Define TBA Loss, optimizer, and run self-supervised pretraining
    # -------------------------------------------------------------------------
    criterion = DINOLoss(
        output_dim=2048,
        warmup_teacher_temp_epochs=5,
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    avg_loss, model = ssl_train(model, dataloader, criterion, optimizer, EPOCHS)

    # -------------------------------------------------------------------------
    # 5. Save the pretrained YOLO backbone for downstream fine-tuning
    # -------------------------------------------------------------------------
    # Reload the YOLO model structure
    yolo = YOLO("models/yolo11n.pt")

    # Transfer the pretrained student backbone weights
    yolo.model.load(model.student_backbone)

    # Save the pretrained model for downstream detection fine-tuning
    yolo.save("models/aft_dino_yolo11n_pretrained.pt")
