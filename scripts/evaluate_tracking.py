import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKEVAL_ROOT = PROJECT_ROOT / "external" / "TrackEval"

sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval


def main():
    eval_config = trackeval.Evaluator.get_default_eval_config()

    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    })

    # Find all prepared validation sequences first
    gt_root = (
        PROJECT_ROOT
        / "results/trackeval/motchallenge/gt/AeroTrack-car-960-val"
    )

    sequence_names = sorted(
        path.name
        for path in gt_root.iterdir()
        if path.is_dir()
    )

    print(f"Evaluating sequences: {sequence_names}")

    dataset_config = (
        trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    )

    dataset_config.update({
        "GT_FOLDER": str(
            PROJECT_ROOT
            / "results/trackeval/motchallenge/gt"
        ),
        "TRACKERS_FOLDER": str(
            PROJECT_ROOT
            / "results/trackeval/motchallenge/trackers"
        ),
        "BENCHMARK": "AeroTrack-car-960",
        "SPLIT_TO_EVAL": "val",
        "TRACKERS_TO_EVAL": ["bytetrack"],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "DO_PREPROC": False,

        "SEQ_INFO": {
            sequence: None
            for sequence in sequence_names
        },
    })

    evaluator = trackeval.Evaluator(eval_config)

    dataset = trackeval.datasets.MotChallenge2DBox(
        dataset_config
    )

    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator.evaluate(
        [dataset],
        metrics,
    )


if __name__ == "__main__":
    main()