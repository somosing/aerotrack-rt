from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/detect/results/training/"
    "yolo26n_visdrone_640_e30/weights/best.pt"
)

SEQUENCES_DIR = Path(
    "data/VisDrone2019-MOT-val/sequences"
)

OUTPUT_DIR = Path(
    "results/tracking/bytetrack_val_960"
)


def process_sequence(sequence_dir: Path):
    # Fresh model = fresh tracker state for each independent video.
    model = YOLO(MODEL_PATH)

    frame_paths = sorted(sequence_dir.glob("*.jpg"))

    output_path = OUTPUT_DIR / f"{sequence_dir.name}.txt"

    rows_written = 0

    with output_path.open("w") as output_file:
        for frame_number, frame_path in enumerate(frame_paths, start=1):

            frame = cv2.imread(str(frame_path))

            if frame is None:
                raise RuntimeError(f"Could not read {frame_path}")

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=960,
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

                # YOLO classes:     0..9
                # VisDrone classes: 1..10
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

    print(
        f"{sequence_dir.name}: "
        f"{len(frame_paths)} frames, "
        f"{rows_written} track observations"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted(
        path
        for path in SEQUENCES_DIR.iterdir()
        if path.is_dir()
    )

    print(f"Sequences found: {len(sequence_dirs)}")
    print()

    for sequence_dir in sequence_dirs:
        process_sequence(sequence_dir)

    print()
    print(f"Saved predictions to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
