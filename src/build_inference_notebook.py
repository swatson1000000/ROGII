"""Generate the Kaggle submission notebook for ROGII.

SKELETON. Mirrors the BirdCLEF pattern: assemble a self-contained inference
notebook (jupyter/) that loads trained artifacts from a Kaggle dataset, predicts
TVT for every scored row of each test well, and writes submission.csv with
columns [id, tvt].

Submission shape (verified): one row per horizontal-well row at/after the
Prediction Start point, id = f"{well_id}_{row_index}".
"""

from __future__ import annotations


def main() -> None:
    # TODO: build the notebook once train.py produces artifacts.
    #   - Load model/artifacts from the Kaggle dataset (no internet at submit time).
    #   - For each test well: predict TVT for rows >= PS, anchored on TVT_input.
    #   - Concatenate to submission.csv (id, tvt) and validate against
    #     data/raw/sample_submission.csv ids.
    raise NotImplementedError("Build after train.py exists — see plan.md.")


if __name__ == "__main__":
    main()
