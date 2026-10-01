from collections import defaultdict
from pathlib import Path


SEQUENCE = "uav0000137_00458_v"

GT_INPUT = Path(
    f"data/VisDrone2019-MOT-val/annotations/{SEQUENCE}.txt"
)

PRED_INPUT = Path(
    f"results/tracking/{SEQUENCE}_bytetrack.txt"
)

OUTPUT_DIR = Path("results/trackeval/car_baseline")

GT_OUTPUT = OUTPUT_DIR / "gt.txt"
PRED_OUTPUT = OUTPUT_DIR / "pred.txt"

VISDRONE_CAR_CLASS = 4
IGNORE_CLASS = 0
IGNORE_OVERLAP_THRESHOLD = 0.5


def intersection_over_detection(det, ignore):
    """Fraction of the detection box covered by an ignored region."""

    dx, dy, dw, dh = det
    ix, iy, iw, ih = ignore

    left = max(dx, ix)
    top = max(dy, iy)
    right = min(dx + dw, ix + iw)
    bottom = min(dy + dh, iy + ih)

    intersection_w = max(0.0, right - left)
    intersection_h = max(0.0, bottom - top)

    intersection = intersection_w * intersection_h
    detection_area = dw * dh

    if detection_area <= 0:
        return 0.0

    return intersection / detection_area


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_gt = []
    ignored_by_frame = defaultdict(list)

    # -------------------------
    # Read VisDrone GT
    # -------------------------

    with GT_INPUT.open() as src:
        for line in src:
            values = line.strip().split(",")

            frame = int(values[0])
            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            score = int(values[6])
            class_id = int(values[7])

            # Explicit VisDrone ignored regions
            if score == 0 and class_id == IGNORE_CLASS:
                ignored_by_frame[frame].append((x, y, w, h))
                continue

            # Car-only experiment
            if score == 1 and class_id == VISDRONE_CAR_CLASS:
                valid_gt.append(
                    (frame, track_id, x, y, w, h)
                )

    # -------------------------
    # Write TrackEval GT
    # -------------------------

    with GT_OUTPUT.open("w") as dst:
        for frame, track_id, x, y, w, h in valid_gt:
            dst.write(
                f"{frame},{track_id},"
                f"{x},{y},{w},{h},"
                f"1,1,1\n"
            )

    # -------------------------
    # Filter predictions
    # -------------------------

    predictions_kept = 0
    predictions_ignored = 0

    with PRED_INPUT.open() as src, PRED_OUTPUT.open("w") as dst:

        for line in src:
            values = line.strip().split(",")

            frame = int(values[0])
            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            confidence = float(values[6])
            class_id = int(values[7])

            # Only evaluate car predictions
            if class_id != VISDRONE_CAR_CLASS:
                continue

            detection = (x, y, w, h)

            should_ignore = False

            for ignore_box in ignored_by_frame.get(frame, []):
                overlap = intersection_over_detection(
                    detection,
                    ignore_box,
                )

                if overlap >= IGNORE_OVERLAP_THRESHOLD:
                    should_ignore = True
                    break

            if should_ignore:
                predictions_ignored += 1
                continue

            dst.write(
                f"{frame},{track_id},"
                f"{x},{y},{w},{h},"
                f"{confidence},-1,-1,-1\n"
            )

            predictions_kept += 1

    print(f"Valid GT car rows:       {len(valid_gt)}")
    print(
        f"Ignored-region boxes:    "
        f"{sum(len(v) for v in ignored_by_frame.values())}"
    )
    print(f"Predictions kept:        {predictions_kept}")
    print(f"Predictions ignored:     {predictions_ignored}")

    print()
    print(f"GT output:   {GT_OUTPUT}")
    print(f"Pred output: {PRED_OUTPUT}")


if __name__ == "__main__":
    main()