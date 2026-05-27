"""
Configuration for SPY data pipeline
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Configuration for data download and processing"""

    # ===== THETADATA CONNECTION =====
    base_url: str = "http://127.0.0.1:25510"
    timeout_seconds: int = 60

    # ===== PATHS =====
    raw_data_path: Path = Path("data/raw")
    merged_data_path: Path = Path("data/merged")

    # ===== SPOT TRADE FILTERS =====
    # All these must match exactly (AND condition)
    spot_ext_condition1: int = 255
    spot_ext_condition2: int = 255
    spot_ext_condition3: int = 255
    spot_ext_condition4: int = 115
    spot_condition: int = 115
    spot_excluded_exchange: int = 57  # Exclude this exchange

    # ===== OPTIONS TRADE FILTERS =====
    # FLAGS must be one of these values (OR condition)
    option_valid_flags: List[int] = None

    def __post_init__(self):
        """Initialize mutable defaults and create directories"""
        if self.option_valid_flags is None:
            self.option_valid_flags = [18, 125, 95]

        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.merged_data_path.mkdir(parents=True, exist_ok=True)

# Global config instance
CONFIG = Config()
