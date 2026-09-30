from pathlib import Path

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


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_gt(path):
    boxes = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            x = int(parts[0])
            y = int(parts[1])
            w = int(parts[2])
            h = int(parts[3])
            class_id = int(parts[5])

            if class_id == 0:
                continue

            boxes.append(
                {
                    "box": (x, y, x + w, y + h),
                    "class_id": class_id,
                }
            )

    return boxes


def main():
    image_path = Path(
        "data/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
    )

    annotation_path = Path(
        "data/VisDrone2019-DET-val/annotations/0000001_02999_d_0000005.txt"
    )

    gt = load_gt(annotation_path)

    model = YOLO("yolo26n.pt")

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False,
    )[0]

    predictions = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        predictions.append(
            {
                "box": (x1, y1, x2, y2),
                "class_id": int(box.cls.item()),
                "confidence": float(box.conf.item()),
            }
        )

    print(f"GT boxes: {len(gt)}")
    print(f"Predictions: {len(predictions)}")
    print()

    for i, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt = None

        for gt_obj in gt:
            iou = box_iou(pred["box"], gt_obj["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_obj

        pred_name = model.names[pred["class_id"]]

        if best_gt is not None:
            gt_name = VISDRONE_CLASSES[best_gt["class_id"]]
        else:
            gt_name = "none"

        print(
            f"{i:02d} | "
            f"pred={pred_name:10s} | "
            f"conf={pred['confidence']:.3f} | "
            f"best IoU={best_iou:.3f} | "
            f"closest GT={gt_name}"
        )


if __name__ == "__main__":
    main()
