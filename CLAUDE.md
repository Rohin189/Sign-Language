# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Camera-only (no gloves, no wearables) isolated word-level sign language classifier for **Indian Sign Language**, built from scratch as a rebuild of an earlier version.

- **Scope:** a small vocabulary of visually distinct signs (`data/raw/` currently holds 11: hello, thank_you, yes, no, please, help, sorry, good, bad, name, namaste). The vocabulary will be scaled up later.
- **Data:** collected by the project owner first (~40 samples per sign); 3-5 additional signers join in a later phase for generalization testing.
- **Stack:** Python 3.11, MediaPipe Hands, OpenCV, NumPy, TensorFlow/Keras. FastAPI and React are planned for deployment.
- **Environment:** conda env `sign_lang_env`, Windows, PowerShell, VS Code.

## Layout

```
data/raw/<sign>/sample_<n>.npy   # raw sequences (git-ignored)
data/processed/                  # preprocessed datasets (git-ignored)
models/                          # trained models (weights git-ignored)
src/config.py                    # single source of truth for all settings
src/utils/landmarks.py           # HandLandmarkExtractor: frame -> 128-dim vector
src/collect_signs.py             # webcam data collection tool
src/preprocess.py                # raw -> data/processed/dataset.npz (stratified split, quality report)
src/train.py                     # baseline LSTM training -> models/baseline_lstm.keras
verify.py                        # local sanity-check script (git-ignored)
```

## config.py (single source of truth)

`src/config.py` defines:
- **`SIGNS`**: the list order defines label encoding. **Never reorder; only append.** During collection it may be temporarily edited to a subset; before any preprocessing or training it must contain the full canonical list matching every folder in `data/raw/`.
- `SEQUENCE_LENGTH = 30`, `NUM_HANDS`, `LANDMARKS_PER_HAND`, `FEATURE_DIM_PER_HAND = 64`, `FEATURE_DIM_TOTAL = 128`.
- Paths `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, all resolved via pathlib from `PROJECT_ROOT`.
- `CAMERA_INDEX`, `SAMPLES_PER_SIGN_PER_PERSON`, `MIN_DETECTION_CONFIDENCE`, `MIN_TRACKING_CONFIDENCE`.

## Feature specification

- Per frame: **128-dim** = 2 hands x (21 landmarks x 3 coords + 1 presence flag).
- Each hand is translated so the wrist (landmark 0) is the origin, then scale-normalized by the wrist-to-middle-MCP distance (landmark 9).
- An absent hand is zero-padded with its presence flag set to 0.
- Hands are slotted **by MediaPipe handedness label** (Left slot, then Right slot), not by detection order.
- Frames are passed to MediaPipe **without horizontal flipping**. Inference must use the same convention, or the left/right slots will swap.
- Each sample is a `(30, 128)` `float32` array saved as `data/raw/<sign>/sample_<n>.npy`. The index continues from existing files, so nothing is overwritten.

## Commands

Activate the environment, then always run scripts **as modules from the project root**:

```powershell
conda activate sign_lang_env
python -m src.collect_signs        # collect data (webcam, owner runs)
python -m src.utils.landmarks      # live landmark extraction test (webcam, owner runs; q to quit)
python -m src.preprocess           # raw -> data/processed/dataset.npz
python -m src.train                # train baseline model
python verify.py                   # inspect a saved sample
```

Never run `python src/<file>.py` directly; it breaks `from src import config` imports.

## Terminal ownership

- Claude runs all terminal work itself: installs, scripts (`preprocess`, `train`, `verify.py`), checks, and git. Do not hand commands to the owner to run.
- Ask the owner to run a command only when it genuinely needs them: interactive webcam or GUI tools (`collect_signs`, `src.utils.landmarks`), or interactive logins. Give the exact command when asking.
- `git commit` and `git push`: Claude runs them, but always asks for confirmation first.

**collect_signs.py controls:** `SPACE` = record, `r` = redo (delete) last sample, `n` = next sign, `q` = quit.

## Conventions and rules

- Use pathlib and `PROJECT_ROOT`-relative paths only; never cwd-relative paths.
- Never hardcode values that belong in `config.py`.
- Validate array shapes before saving or training; discard malformed samples.
- Do not commit `data/raw`, `data/processed`, virtual environments, or large model weights. `.gitignore` must cover them.
- Known harmless warning: protobuf `SymbolDatabase.GetPrototype()` deprecation message from MediaPipe.
- If `cv2.VideoCapture(0)` fails, try camera index 1 (external USB camera) via `CAMERA_INDEX` in `config.py`.

## Git workflow

- Commit and push only at the end of a completed phase, never per day or per file.
- Commit message format: `Phase <N>: <concise summary>`.
- Before any commit, run `git status` and confirm no data or environment files are staged.

## Phase roadmap

1. **Data foundation** (complete): env, config, landmark extraction, collection script, dataset collected.
2. Preprocessing and baseline model (`preprocess.py`, `train.py`). *(in progress: both written; baseline test accuracy 0.939, pending owner review)*
3. Model improvement and evaluation. *(upcoming)*
4. Real-time inference. *(upcoming)*
5. Multi-signer expansion and generalization testing. *(upcoming)*
6. Deployment: FastAPI backend, React frontend. *(upcoming)*

## How to work with the owner

- Formal, professional tone.
- Explain the concept and the reasoning behind a decision before presenting code.
- Proceed topic by topic, in structured steps.
- When debugging, improve the existing code iteratively rather than replacing it wholesale.
- Where practical, let the owner attempt the implementation first, then review it.
- Be specific: give exact file paths and expected output; run commands rather than handing them over (see Terminal ownership).
