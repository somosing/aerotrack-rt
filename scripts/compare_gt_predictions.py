from pathlib import Path

import cv2
from ultralytics import YOLO


VISDRONE_CLASSES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def load_visdrone_annotations(path):
    annotations = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            x = int(parts[0])
            y = int(parts[1])
            w = int(parts[2])
            h = int(parts[3])
            score = int(parts[4])
            class_id = int(parts[5])

            annotations.append(
                {
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "score": score,
                    "class_id": class_id,
                }
            )

    return annotations


def main():
    image_path = Path(
        "data/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
    )

    annotation_path = Path(
        "data/VisDrone2019-DET-val/annotations/0000001_02999_d_0000005.txt"
    )

    output_dir = Path("results/detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read {image_path}")

    # -------------------------
    # Ground truth
    # -------------------------
    gt = load_visdrone_annotations(annotation_path)

    for ann in gt:
        class_id = ann["class_id"]

        # Skip ignored regions
        if class_id == 0:
            continue

        x1 = ann["x1"]
        y1 = ann["y1"]
        x2 = ann["x2"]
        y2 = ann["y2"]

        label = VISDRONE_CLASSES.get(class_id, str(class_id))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"GT {label}",
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # -------------------------
    # Predictions
    # -------------------------
    model = YOLO("yolo26n.pt")

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False,
    )[0]

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )

        label = model.names[class_id]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            f"PRED {label} {confidence:.2f}",
            (x1, min(y2 + 14, image.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_path = output_dir / "gt_vs_predictions.jpg"

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise RuntimeError(f"Could not save {output_path}")

    print(f"Ground-truth annotations: {len(gt)}")
    print(f"Predictions: {len(result.boxes)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
