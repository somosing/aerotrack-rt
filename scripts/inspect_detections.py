from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    image_dir = Path("data/VisDrone2019-DET-val/images")
    output_dir = Path("results/detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))

    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    image_path = images[0]

    print(f"Image: {image_path}")

    model = YOLO("yolo26n.pt")

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False,
    )

    result = results[0]

    print(f"Image shape: {result.orig_shape}")
    print(f"Detections: {len(result.boxes)}")
    print()

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width = x2 - x1
        height = y2 - y1

        print(
            f"{i:03d} | "
            f"class={model.names[cls_id]:12s} | "
            f"conf={confidence:.3f} | "
            f"size={width:.1f}x{height:.1f}px | "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

    annotated = result.plot()

    output_path = output_dir / "first_frame.jpg"
    cv2.imwrite(str(output_path), annotated)

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
