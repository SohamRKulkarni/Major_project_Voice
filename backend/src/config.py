import os
from pathlib import Path

# Project root (goes 2 levels up from this file: backend/src/config.py -> backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Language-specific directories
ENGLISH_DIR = RAW_DATA_DIR / "english"
HINDI_DIR = RAW_DATA_DIR / "hindi"

STRESS_LEVELS = ["no_stress", "low_stress", "medium_stress", "high_stress"]

# Create directories if they don't exist
for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    ENGLISH_DIR,
    HINDI_DIR,
]:
    os.makedirs(path, exist_ok=True)

for lang_dir in [ENGLISH_DIR, HINDI_DIR]:
    for level in STRESS_LEVELS:
        os.makedirs(lang_dir / level, exist_ok=True)

print("✅ Project directories verified and ready.")
