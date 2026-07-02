from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_PARTITIONED_DIR = DATA_DIR / "cache_partitioned"

# Debug helper
def debug_paths():
    return {
        "BASE_DIR": str(BASE_DIR),
        "DATA_DIR": str(DATA_DIR),
        "CACHE_PARTITIONED_DIR": str(CACHE_PARTITIONED_DIR),
        "CACHE_EXISTS": CACHE_PARTITIONED_DIR.exists(),
    }