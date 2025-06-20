import pandas as pd
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_features(symbol, timeframes, data_dir, features_dir, config):
    logging.info("Generating features on a consistent, live-compatible column set...")
    base_tf_data = None

    core_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    for tf in tqdm(timeframes, desc="Processing Timeframes for Features"):
        file_path = os.path.join(data_dir, f"{symbol.replace('/', '_')}_{tf}.csv")
        if not os.path.exists(file_path):
            logging.warning(f"Data for {tf} not found. Skipping.")
            continue

        df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)

        df = df[[col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume']]]

        df.ta.rsi(length=config['rsi_length'], append=True)
        df.ta.macd(fast=config['macd_fast'], slow=config['macd_slow'], signal=config['macd_signal'], append=True)
        df.ta.bbands(length=config['bbands_length'], std=config['bbands_std'], append=True)
        df.ta.obv(append=True)

        df = df.add_suffix(f"_{tf}")

        if base_tf_data is None:
            base_tf_data = df
        else:
            base_tf_data = pd.merge(base_tf_data, df, on="timestamp", how="left")

    base_tf_data.ffill(inplace=True)
    base_tf_data.dropna(inplace=True)

    price_col = f'close_{timeframes[0]}'
    base_tf_data['future_price'] = base_tf_data[price_col].shift(-config['target_horizon'])
    base_tf_data['target'] = (base_tf_data['future_price'] > base_tf_data[price_col]).astype(int)
    base_tf_data.dropna(subset=['future_price', 'target'], inplace=True)
    base_tf_data.drop(columns=['future_price'], inplace=True)

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    output_path = os.path.join(features_dir, f"{symbol.replace('/', '_')}_features.csv")
    base_tf_data.to_csv(output_path)
    logging.info(f"Features saved to {output_path}")
    return base_tf_data