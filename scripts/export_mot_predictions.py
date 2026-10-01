from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/detect/results/training/"
    "yolo26n_visdrone_640_e30/weights/best.pt"
)

SEQUENCE_DIR = Path(
    "data/VisDrone2019-MOT-val/sequences/"
    "uav0000137_00458_v"
)

OUTPUT_PATH = Path(
    "results/tracking/"
    "uav0000137_00458_v_bytetrack.txt"
)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    frame_paths = sorted(SEQUENCE_DIR.glob("*.jpg"))

    print(f"Frames: {len(frame_paths)}")

    rows_written = 0

    with OUTPUT_PATH.open("w") as output_file:

        for frame_number, frame_path in enumerate(frame_paths, start=1):

            frame = cv2.imread(str(frame_path))

            if frame is None:
                raise RuntimeError(f"Could not read {frame_path}")

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=640,
                conf=0.1,
                device=0,
                verbose=False,
            )[0]

            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            for box, track_id, confidence, class_id in zip(
                boxes,
                track_ids,
                confidences,
                class_ids,
            ):
                x1, y1, x2, y2 = box

                width = x2 - x1
                height = y2 - y1

                visdrone_class_id = class_id + 1

                output_file.write(
                    f"{frame_number},"
                    f"{track_id},"
                    f"{x1:.2f},"
                    f"{y1:.2f},"
                    f"{width:.2f},"
                    f"{height:.2f},"
                    f"{confidence:.4f},"
                    f"{visdrone_class_id},"
                    f"-1,"
                    f"-1\n"
                )

                rows_written += 1

    print(f"Rows written: {rows_written}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
