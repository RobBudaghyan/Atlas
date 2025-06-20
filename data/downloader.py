import os
import pandas as pd
from binance.client import Client
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_binance_data(symbol, timeframes, start_date, data_dir):
    """
    Downloads historical OHLCV data for a given symbol and timeframes from Binance.
    """
    client = Client()
    for tf in tqdm(timeframes, desc="Downloading Timeframes"):
        logging.info(f"Downloading {symbol} data for {tf} timeframe...")
        try:
            klines_generator = client.get_historical_klines_generator(
                symbol.replace("/", ""), tf, start_str=start_date
            )

            klines = list(tqdm(klines_generator, desc=f"Fetching {tf} klines"))

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Save data
            output_path = os.path.join(data_dir, f"{symbol.replace('/', '_')}_{tf}.csv")
            df.to_csv(output_path)
            logging.info(f"Saved {symbol} data for {tf} to {output_path}")

        except Exception as e:
            logging.error(f"Failed to download data for {tf}: {e}")


if __name__ == '__main__':
    from config import SYMBOL, TIMEFRAMES, DATA_START_DATE, DATA_DIRECTORY

    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
    download_binance_data(SYMBOL, TIMEFRAMES, DATA_START_DATE, DATA_DIRECTORY)