# src/preprocess.py
"""
Builds the training dataset from raw collected samples.

    data/raw/<sign>/sample_<n>.npy  ->  data/processed/dataset.npz

Steps:
  1. Load every sample for each sign in config.SIGNS (label = index in SIGNS).
  2. Validate shape / dtype / NaNs; skip malformed samples.
  3. Report data quality (frames with no hands detected).
  4. Stratified train / val / test split with a fixed seed.
  5. Save all splits plus the sign list into a single .npz file.

Run from the project root:
    python -m src.preprocess
"""

import re
from pathlib import Path

import numpy as np

from src import config

EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FEATURE_DIM_TOTAL)

# Presence-flag positions within the 128-dim frame vector (last value of each hand block)
LEFT_FLAG_IDX = config.FEATURE_DIM_PER_HAND - 1          # 63
RIGHT_FLAG_IDX = config.FEATURE_DIM_TOTAL - 1            # 127


def _sample_number(path: Path) -> int:
    """Extracts n from 'sample_<n>.npy' so files sort numerically, not alphabetically."""
    match = re.search(r"sample_(\d+)\.npy$", path.name)
    return int(match.group(1)) if match else -1


def check_sign_folders() -> None:
    """Warns about folders in data/raw that are not part of config.SIGNS."""
    on_disk = {p.name for p in config.RAW_DATA_DIR.iterdir() if p.is_dir()}
    unused = sorted(on_disk - set(config.SIGNS))
    if unused:
        print(f"⚠️  Folders in data/raw not in config.SIGNS (ignored): {unused}")


def is_valid_sample(sample: np.ndarray) -> bool:
    return (
        sample.shape == EXPECTED_SHAPE
        and sample.dtype == np.float32
        and not np.isnan(sample).any()
    )


def load_dataset():
    """
    Returns:
        X       (N, SEQUENCE_LENGTH, FEATURE_DIM_TOTAL) float32
        y       (N,) int64 label indices into config.SIGNS
        sources (N,) str  'sign/sample_n.npy' for traceability
    """
    X, y, sources = [], [], []
    skipped = 0

    for label, sign in enumerate(config.SIGNS):
        sign_dir = config.RAW_DATA_DIR / sign
        files = sorted(sign_dir.glob("sample_*.npy"), key=_sample_number) if sign_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No samples found for sign '{sign}' in {sign_dir}")

        for f in files:
            sample = np.load(f)
            if not is_valid_sample(sample):
                print(f"⚠️  Skipping malformed sample {sign}/{f.name}: "
                      f"shape={sample.shape}, dtype={sample.dtype}")
                skipped += 1
                continue
            X.append(sample)
            y.append(label)
            sources.append(f"{sign}/{f.name}")

    print(f"Loaded {len(X)} samples across {len(config.SIGNS)} signs ({skipped} skipped).")
    return np.stack(X), np.array(y, dtype=np.int64), np.array(sources)


def report_quality(X: np.ndarray, y: np.ndarray, sources: np.ndarray) -> None:
    """Prints, per sign, how often no hand was detected. Flags weak samples; does not remove them."""
    no_hand = (X[:, :, LEFT_FLAG_IDX] == 0) & (X[:, :, RIGHT_FLAG_IDX] == 0)  # (N, T)
    empty_ratio = no_hand.mean(axis=1)                                        # (N,)

    print("\nData quality (share of frames with no hand detected):")
    for label, sign in enumerate(config.SIGNS):
        mask = y == label
        print(f"  {sign:12s} mean={empty_ratio[mask].mean():6.1%}  worst={empty_ratio[mask].max():6.1%}")

    weak = np.where(empty_ratio > config.MAX_EMPTY_FRAME_RATIO)[0]
    if len(weak):
        print(f"\n⚠️  {len(weak)} sample(s) have >{config.MAX_EMPTY_FRAME_RATIO:.0%} empty frames "
              f"(consider re-recording):")
        for i in weak:
            print(f"     {sources[i]}  ({empty_ratio[i]:.0%} empty)")


def stratified_split(y: np.ndarray):
    """
    Splits indices per class so every sign keeps the same proportions in each split.
    Returns (train_idx, val_idx, test_idx).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    train_idx, val_idx, test_idx = [], [], []

    for label in range(config.NUM_CLASSES):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)

        n_test = max(1, round(len(idx) * config.TEST_SPLIT))
        n_val = max(1, round(len(idx) * config.VAL_SPLIT))
        if len(idx) - n_test - n_val < 1:
            raise ValueError(f"Sign '{config.SIGNS[label]}' has too few samples ({len(idx)}) to split.")

        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    # Shuffle within each split so classes are interleaved
    splits = [np.array(s, dtype=np.int64) for s in (train_idx, val_idx, test_idx)]
    for s in splits:
        rng.shuffle(s)

    # Sanity check: no sample appears in more than one split
    a, b, c = (set(s.tolist()) for s in splits)
    assert not (a & b or a & c or b & c), "Split overlap detected"
    assert len(a | b | c) == len(y), "Some samples were not assigned to a split"
    return splits


def print_split_summary(y: np.ndarray, splits) -> None:
    names = ("train", "val", "test")
    print(f"\n{'sign':12s} " + " ".join(f"{n:>6s}" for n in names))
    for label, sign in enumerate(config.SIGNS):
        counts = [int((y[s] == label).sum()) for s in splits]
        print(f"{sign:12s} " + " ".join(f"{c:6d}" for c in counts))
    print(f"{'TOTAL':12s} " + " ".join(f"{len(s):6d}" for s in splits))


def main():
    check_sign_folders()
    X, y, sources = load_dataset()
    report_quality(X, y, sources)

    train_idx, val_idx, test_idx = stratified_split(y)
    print_split_summary(y, (train_idx, val_idx, test_idx))

    np.savez_compressed(
        config.DATASET_PATH,
        X_train=X[train_idx], y_train=y[train_idx], src_train=sources[train_idx],
        X_val=X[val_idx],     y_val=y[val_idx],     src_val=sources[val_idx],
        X_test=X[test_idx],   y_test=y[test_idx],   src_test=sources[test_idx],
        signs=np.array(config.SIGNS),
    )
    print(f"\n✅ Saved dataset to {config.DATASET_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
