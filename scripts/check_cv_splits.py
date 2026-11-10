"""
Verify 3-fold group-aware cross-validation for AqSolDB.

Only the original training and validation subsets are used.
The held-out test set remains excluded from cross-validation.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

from scripts.benchmark_models import prepare_dataset


def main():
    """Check fold membership and molecular-group isolation."""

    df = prepare_dataset()

    # Reserve the original test set for final evaluation.
    dev = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)
    test = df[df["split"] == "test"]

    dev_groups = set(dev["group_key"])
    test_groups = set(test["group_key"])

    assert dev_groups.isdisjoint(test_groups)

    # Fixed random seed ensures reproducible fold assignments.
    cv = GroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    validation_counts = np.zeros(len(dev), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(dev, groups=dev["group_key"]), start=1
    ):
        train_groups = set(dev.iloc[train_idx]["group_key"])
        val_groups = set(dev.iloc[val_idx]["group_key"])

        # A molecular group must never appear on both sides.
        assert train_groups.isdisjoint(val_groups)

        validation_counts[val_idx] += 1

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"validation={len(val_idx)}, "
            "group overlap=0"
        )

    # Every development molecule must be validated exactly once.
    assert np.all(validation_counts == 1)

    print("Development molecules:", len(dev))
    print("Held-out test molecules:", len(test))
    print("PASS: All three CV folds verified!")


if __name__ == "__main__":
    main()
