import argparse
import logging
from config import *
from data.downloader import download_binance_data
from features.generator import generate_features
from models.trainer import train_model
from backtest.engine import run_backtest
from testnet.trader import run_testnet_trader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="ETH/USDT Futures ML Trading Bot")
    parser.add_argument("action", choices=["download", "features", "train", "backtest", "trade"])
    args = parser.parse_args()

    if args.action == "download":
        logging.info("Starting data download...")
        download_binance_data(SYMBOL, TIMEFRAMES, DATA_START_DATE, DATA_DIRECTORY)

    elif args.action == "features":
        logging.info("Starting feature generation...")
        generate_features(SYMBOL, TIMEFRAMES, DATA_DIRECTORY, FEATURES_DIRECTORY, FEATURE_GENERATION_CONFIG)

    elif args.action == "train":
        logging.info("Starting model training...")
        train_model(SYMBOL, FEATURES_DIRECTORY, MODEL_DIRECTORY, WALK_FORWARD_VALIDATION)

    elif args.action == "backtest":
        logging.info("Starting backtest...")
        BACKTESTING_CONFIG['signal_persistence'] = 3
        BACKTESTING_CONFIG['cooldown_minutes'] = 15
        run_backtest(SYMBOL, FEATURES_DIRECTORY, MODEL_DIRECTORY, BACKTESTING_CONFIG)

    elif args.action == "trade":
        logging.info("Starting Testnet trading...")
        full_config = {
            "TESTNET_CONFIG": TESTNET_CONFIG,
            "FEATURE_GENERATION_CONFIG": FEATURE_GENERATION_CONFIG,
            "TIMEFRAMES": TIMEFRAMES,
            "BINANCE_API_KEY": BINANCE_API_KEY,
            "BINANCE_API_SECRET": BINANCE_API_SECRET,
            "BACKTESTING_CONFIG": BACKTESTING_CONFIG,
        }
        run_testnet_trader(full_config, MODEL_DIRECTORY)


if __name__ == "__main__":
    main()