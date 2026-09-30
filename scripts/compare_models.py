from pathlib import Path

from ultralytics import YOLO


IMAGE = Path(
    "data/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
)

GENERIC_MODEL = "yolo26n.pt"

FINETUNED_MODEL = (
    "runs/detect/results/training/"
    "yolo26n_visdrone_640_e30/weights/best.pt"
)


def run_model(model_path, name):
    model = YOLO(model_path)

    result = model.predict(
        source=str(IMAGE),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False,
    )[0]

    print(f"{name}")
    print(f"Detections: {len(result.boxes)}")

    counts = {}

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]

        counts[class_name] = counts.get(class_name, 0) + 1

    for class_name, count in sorted(counts.items()):
        print(f"  {class_name:18s}: {count}")

    print()


def main():
    run_model(GENERIC_MODEL, "Generic COCO YOLO26n")
    run_model(FINETUNED_MODEL, "VisDrone fine-tuned YOLO26n")


if __name__ == "__main__":
    main()
