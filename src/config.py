# src/config.py

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/config.py -> project root

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist (safe to call every run)
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sign vocabulary ─────────────────────────────────────────────
# Replace with your own 10 signs if these don't match your plan
SIGNS = [
    #"hello",
    #"thank_you",
    "yes",
    "no",
    "please",
    "help",
    "sorry",
    "good",
    "bad",
    "name",
]

NUM_CLASSES = len(SIGNS)

# ── Sequence & feature dimensions ──────────────────────────────
SEQUENCE_LENGTH = 30          # frames per sample

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3       # x, y, z
PRESENCE_FLAG_DIM = 1         # 1 = hand detected, 0 = absent

FEATURE_DIM_PER_HAND = (LANDMARKS_PER_HAND * COORDS_PER_LANDMARK) + PRESENCE_FLAG_DIM  # 64
FEATURE_DIM_TOTAL = FEATURE_DIM_PER_HAND * NUM_HANDS  # 128

# ── Data collection settings ────────────────────────────────────
SAMPLES_PER_SIGN_PER_PERSON = 40   # target samples per sign, per contributor
CAMERA_INDEX = 0                   # default webcam

# ── MediaPipe settings ──────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5