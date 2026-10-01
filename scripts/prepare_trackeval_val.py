from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SEQUENCES_DIR = Path("data/VisDrone2019-MOT-val/sequences")
ANNOTATIONS_DIR = Path("data/VisDrone2019-MOT-val/annotations")
PREDICTIONS_DIR = Path("results/tracking/bytetrack_val_960")

GT_OUTPUT_ROOT = Path(
    "results/trackeval/motchallenge/gt/AeroTrack-car-960-val"
)

PRED_OUTPUT_DIR = Path(
    "results/trackeval/motchallenge/"
    "trackers/AeroTrack-car-960-val/bytetrack/data"
)

CAR_CLASS = 4
IGNORE_CLASSES = {0, 11}
IGNORE_THRESHOLD = 0.5


def make_ignore_mask(boxes, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    for x, y, w, h in boxes:
        x = max(0, int(round(x)))
        y = max(0, int(round(y)))
        w = max(1, int(round(w)))
        h = max(1, int(round(h)))

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x < x2 and y < y2:
            mask[y:y2, x:x2] = 1

    return mask


def ignore_fraction(box, mask):
    x, y, w, h = box

    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))

    height, width = mask.shape

    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x >= x2 or y >= y2:
        return 0.0

    ignored_pixels = mask[y:y2, x:x2].sum()

    return ignored_pixels / float(w * h)


def process_sequence(sequence_dir):
    sequence = sequence_dir.name

    gt_input = ANNOTATIONS_DIR / f"{sequence}.txt"
    pred_input = PREDICTIONS_DIR / f"{sequence}.txt"

    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    num_frames = len(frame_paths)

    first_frame = cv2.imread(str(frame_paths[0]))

    if first_frame is None:
        raise RuntimeError(f"Could not read {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    ignore_by_frame = defaultdict(list)
    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)

    # -----------------------
    # Read GT
    # -----------------------

    with gt_input.open() as file:
        for line in file:
            values = line.strip().split(",")

            frame = int(values[0])
            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            score = int(values[6])
            class_id = int(values[7])

            if class_id in IGNORE_CLASSES:
                ignore_by_frame[frame].append((x, y, w, h))

            elif score == 1 and class_id == CAR_CLASS:
                gt_by_frame[frame].append(
                    (track_id, x, y, w, h)
                )

    # -----------------------
    # Read predictions
    # -----------------------

    with pred_input.open() as file:
        for line in file:
            values = line.strip().split(",")

            frame = int(values[0])
            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            confidence = float(values[6])
            class_id = int(values[7])

            if class_id == CAR_CLASS:
                pred_by_frame[frame].append(
                    (track_id, x, y, w, h, confidence)
                )

    gt_output_dir = GT_OUTPUT_ROOT / sequence
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    pred_output_dir = PRED_OUTPUT_DIR
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    gt_output = gt_output_dir / "gt" / "gt.txt"
    gt_output.parent.mkdir(parents=True, exist_ok=True)

    pred_output = pred_output_dir / f"{sequence}.txt"

    gt_raw = sum(len(v) for v in gt_by_frame.values())
    pred_raw = sum(len(v) for v in pred_by_frame.values())

    gt_kept = 0
    pred_kept = 0

    with gt_output.open("w") as gt_file, pred_output.open("w") as pred_file:

        for frame in range(1, num_frames + 1):

            ignore_mask = make_ignore_mask(
                ignore_by_frame.get(frame, []),
                height,
                width,
            )

            for track_id, x, y, w, h in gt_by_frame.get(frame, []):

                if ignore_fraction((x, y, w, h), ignore_mask) >= IGNORE_THRESHOLD:
                    continue

                gt_file.write(
                    f"{frame},{track_id},"
                    f"{x},{y},{w},{h},"
                    f"1,1,1\n"
                )

                gt_kept += 1

            for track_id, x, y, w, h, confidence in pred_by_frame.get(frame, []):

                if ignore_fraction((x, y, w, h), ignore_mask) >= IGNORE_THRESHOLD:
                    continue

                pred_file.write(
                    f"{frame},{track_id},"
                    f"{x},{y},{w},{h},"
                    f"{confidence},-1,-1,-1\n"
                )

                pred_kept += 1

    seqinfo = gt_output_dir / "seqinfo.ini"

    seqinfo.write_text(
        "[Sequence]\n"
        f"name={sequence}\n"
        "imDir=img1\n"
        "frameRate=25\n"
        f"seqLength={num_frames}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )

    ignored_boxes = sum(
        len(v) for v in ignore_by_frame.values()
    )

    print(
        f"{sequence}: "
        f"GT {gt_raw}->{gt_kept}, "
        f"Pred {pred_raw}->{pred_kept}, "
        f"ignore_boxes={ignored_boxes}"
    )


def main():
    sequence_dirs = sorted(
        path
        for path in SEQUENCES_DIR.iterdir()
        if path.is_dir()
    )

    print(f"Sequences: {len(sequence_dirs)}")
    print()

    for sequence_dir in sequence_dirs:
        process_sequence(sequence_dir)


if __name__ == "__main__":
    main()
