import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

SYMBOL = "ETH/USDT"
TIMEFRAMES = ["1m", "5m", "15m", "1h"]

DATA_START_DATE = "2022-01-01"
DATA_DIRECTORY = "data"

FEATURE_GENERATION_CONFIG = {
    "rsi_length": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bbands_length": 20,
    "bbands_std": 2,
    "target_horizon": 5,
}
FEATURES_DIRECTORY = "features"

MODEL_TYPE = "lightgbm"
MODEL_DIRECTORY = "models"
WALK_FORWARD_VALIDATION = {
    "train_size": 365 * 24 * 60,
    "test_size": 30 * 24 * 60,
}

BACKTESTING_CONFIG = {
    "slippage": 0.0002,
    "fees": 0.0005,
    "initial_capital": 10000,
    "position_sizing": 0.1,
}

TESTNET_CONFIG = {
    "symbol": "ETHUSDT",
    "leverage": 5,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.06,
    "polling_interval_seconds": 60,
}