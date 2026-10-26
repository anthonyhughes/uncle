import os
from pathlib import Path

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", None)
if DATA_DIRECTORY is None:
    DATASETS_PATH = Path(__file__).parent
else:
    DATASETS_PATH = Path(DATA_DIRECTORY) / "datasets"

if DATA_DIRECTORY is None:
    RAW_DATA_PATH = Path(__file__).parent / "raw_data"
else:
    RAW_DATA_PATH = Path(DATA_DIRECTORY) / "datasets" / "raw_data"
