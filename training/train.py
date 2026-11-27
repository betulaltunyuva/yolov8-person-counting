"""Step 2: fine-tune YOLO26n on the custom person dataset.

These are the exact settings stored inside `models/best.pt`. The original
run was done on a Google Colab GPU; on a CPU it is much slower.

Usage (from the project root):
    python training/train.py
"""

from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    # Start from the COCO-pretrained YOLO26 nano checkpoint (transfer learning)
    model = YOLO("yolo26n.pt")

    model.train(
        data=str(PROJECT_DIR / "dataset" / "data.yaml"),
        epochs=50,
        batch=16,
        imgsz=640,
        project=str(PROJECT_DIR / "runs"),
        name="human_counting_v2",
    )

    # Evaluate the best weights on the held-out test split
    best = YOLO(str(PROJECT_DIR / "runs" / "human_counting_v2" / "weights" / "best.pt"))
    best.val(data=str(PROJECT_DIR / "dataset" / "data.yaml"), split="test")
