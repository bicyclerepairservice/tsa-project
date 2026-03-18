from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "src" / "data" / "m4_monthly_dataset.tsf"

@dataclass
class TSOptimizationConfig:
    history: int = 2 * 12 # 2 года
    horizon: int = 1 * 12 # 1 год
    start_train_size: int = 6 * 12 # 6 лет
    step_size: int = 1  # 1 месяц
    season_len: int = 12  # Годовая сезонность

TS_OPTIMIZATION_CONFIG = TSOptimizationConfig()