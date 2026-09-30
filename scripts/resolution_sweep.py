from pathlib import Path
from time import perf_counter

import torch
from ultralytics import YOLO


IMAGE = Path(
    "data/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
)

WARMUP_RUNS = 5
MEASURE_RUNS = 30


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    model = YOLO("yolo26n.pt")

    for imgsz in [640, 960, 1280]:

        # Warm up GPU
        for _ in range(WARMUP_RUNS):
            model.predict(
                source=str(IMAGE),
                imgsz=imgsz,
                conf=0.25,
                device=0,
                verbose=False,
            )

        synchronize()

        times_ms = []
        detections = None

        for _ in range(MEASURE_RUNS):
            synchronize()
            start = perf_counter()

            result = model.predict(
                source=str(IMAGE),
                imgsz=imgsz,
                conf=0.25,
                device=0,
                verbose=False,
            )[0]

            synchronize()
            end = perf_counter()

            times_ms.append((end - start) * 1000.0)
            detections = len(result.boxes)

        avg_ms = sum(times_ms) / len(times_ms)
        fps = 1000.0 / avg_ms

        print(
            f"imgsz={imgsz:4d} | "
            f"detections={detections:3d} | "
            f"avg={avg_ms:7.2f} ms | "
            f"FPS={fps:6.1f}"
        )


if __name__ == "__main__":
    main()
