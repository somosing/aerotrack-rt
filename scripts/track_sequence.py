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

OUTPUT_DIR = Path("results/tracking/bytetrack_baseline")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    frame_paths = sorted(SEQUENCE_DIR.glob("*.jpg"))

    print(f"Frames: {len(frame_paths)}")

    for frame_index, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            raise RuntimeError(f"Could not read {frame_path}")

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=640,
            conf=0.1,
            device=0,
            verbose=False,
        )

        result = results[0]

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = []

        annotated = result.plot()

        output_path = OUTPUT_DIR / frame_path.name
        cv2.imwrite(str(output_path), annotated)

        print(
            f"frame={frame_index:03d} "
            f"detections={len(result.boxes):3d} "
            f"tracked={len(track_ids):3d}"
        )


if __name__ == "__main__":
    main()
