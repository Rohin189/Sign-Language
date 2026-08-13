# src/collect_signs.py

from pathlib import Path

import cv2
import numpy as np

from src import config
from src.utils.landmarks import HandLandmarkExtractor


def get_next_sample_index(sign_dir: Path) -> int:
    """
    Looks at existing sample_<n>.npy files in a sign's folder and returns
    the next index to use, so re-running the script never overwrites data.
    """
    existing = list(sign_dir.glob("sample_*.npy"))
    if not existing:
        return 0
    indices = [int(f.stem.split("_")[1]) for f in existing]
    return max(indices) + 1


def delete_last_sample(sign_dir: Path, current_idx: int) -> int:
    """
    Deletes sample_<current_idx - 1>.npy if it exists and returns the updated index.
    """
    if current_idx <= 0:
        print("⚠️  No samples to redo for this sign yet!")
        return current_idx

    target_idx = current_idx - 1
    last_sample_path = sign_dir / f"sample_{target_idx}.npy"

    if last_sample_path.exists():
        last_sample_path.unlink()
        print(f"🗑️  Deleted: {last_sample_path.name} — ready to record again.")
        return target_idx
    else:
        print(f"⚠️  File {last_sample_path.name} not found.")
        return current_idx


def collect_for_sign(sign_name: str, extractor: HandLandmarkExtractor, cap: cv2.VideoCapture):
    sign_dir = config.RAW_DATA_DIR / sign_name
    sign_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = get_next_sample_index(sign_dir)
    print(f"\n--- Collecting for sign: '{sign_name}' ---")
    print(f"Existing samples: {sample_idx}.")
    print("Controls: SPACE=record | r=redo last | n=next sign | q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Sign: {sign_name} | Samples: {sample_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE=record  r=redo last  n=next sign  q=quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Data Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return "quit"
        elif key == ord('n'):
            return "next"
        elif key == ord('r'):
            sample_idx = delete_last_sample(sign_dir, sample_idx)
        elif key == ord(' '):
            recorded = record_one_sequence(sign_dir, sample_idx, extractor, cap)
            if recorded:
                sample_idx += 1


def record_one_sequence(sign_dir: Path, sample_idx: int, extractor: HandLandmarkExtractor, cap: cv2.VideoCapture) -> bool:
    """
    Records exactly SEQUENCE_LENGTH frames with a short countdown first.
    Returns True if successfully saved, False otherwise.
    """
    for countdown in [3, 2, 1]:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"Starting in {countdown}...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(500)

    sequence = []
    for frame_num in range(config.SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            break

        feature_vector, results = extractor.process_frame(frame)
        sequence.append(feature_vector)

        display_frame = frame.copy()
        cv2.putText(display_frame, f"RECORDING frame {frame_num + 1}/{config.SEQUENCE_LENGTH}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Data Collection", display_frame)
        cv2.waitKey(1)

    sequence = np.array(sequence, dtype=np.float32)

    if sequence.shape != (config.SEQUENCE_LENGTH, config.FEATURE_DIM_TOTAL):
        print(f"WARNING: sample shape {sequence.shape} != expected "
              f"({config.SEQUENCE_LENGTH}, {config.FEATURE_DIM_TOTAL}) — discarding.")
        return False

    save_path = sign_dir / f"sample_{sample_idx}.npy"
    np.save(save_path, sequence)
    print(f"Saved: {save_path.name}  (shape={sequence.shape})")
    return True


def main():
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"ERROR: could not open camera index {config.CAMERA_INDEX}. Try CAMERA_INDEX = 1 in config.py.")
        return

    sign_index = 0
    while sign_index < len(config.SIGNS):
        sign_name = config.SIGNS[sign_index]
        result = collect_for_sign(sign_name, extractor, cap)

        if result == "quit":
            break
        elif result == "next":
            sign_index += 1

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("\nCollection session ended.")


if __name__ == "__main__":
    main()