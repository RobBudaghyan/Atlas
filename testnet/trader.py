import ccxt
import pandas as pd
import pandas_ta as ta
import time
import logging
import joblib
import os
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_live_features(binance, symbol, timeframes, feature_config, feature_names):
    base_tf_data = None
    for tf in timeframes:
        candles = binance.fetch_ohlcv(symbol, timeframe=tf, limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.ta.rsi(length=feature_config['rsi_length'], append=True)
        df.ta.macd(fast=feature_config['macd_fast'], slow=feature_config['macd_slow'],
                   signal=feature_config['macd_signal'], append=True)
        df.ta.bbands(length=feature_config['bbands_length'], std=feature_config['bbands_std'], append=True)
        df.ta.obv(append=True)
        df = df.add_suffix(f"_{tf}")
        df.set_index(f'timestamp_{tf}', inplace=True)
        if base_tf_data is None:
            base_tf_data = df
        else:
            base_tf_data = pd.merge(base_tf_data, df, left_index=True, right_index=True, how="left")
    base_tf_data.ffill(inplace=True)
    base_tf_data.dropna(inplace=True)
    if base_tf_data.empty:
        return None
    latest_features = base_tf_data.iloc[-2]
    if not all(f in latest_features.index for f in feature_names):
        return None
    return latest_features[feature_names]


def run_testnet_trader(config, model_dir):
    logging.info("Connecting to Binance Testnet...")

    testnet_config = config['TESTNET_CONFIG']
    feature_config = config['FEATURE_GENERATION_CONFIG']
    timeframes = config['TIMEFRAMES']
    backtest_config = config['BACKTESTING_CONFIG']

    binance = ccxt.binance({
        'apiKey': config['BINANCE_API_KEY'],
        'secret': config['BINANCE_API_SECRET'],
        'options': {'defaultType': 'future'},
    })
    binance.set_sandbox_mode(True)

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_model_path = os.path.join(model_dir, model_files[-1])
    model = joblib.load(latest_model_path)
    feature_names = model.feature_name_
    logging.info(f"Loaded latest model: {latest_model_path}")

    persistence = backtest_config.get('signal_persistence', 3)
    recent_signals = deque(maxlen=persistence)
    was_in_trade_state = False

    while True:
        try:
            logging.info("--- New Cycle ---")

            positions = binance.fetch_positions([testnet_config['symbol']])
            current_position = next(
                (p for p in positions if p['symbol'] == testnet_config['symbol'] and p['side'] == 'long'), None)
            position_amount = float(current_position['contracts']) if current_position and current_position[
                'contracts'] is not None else 0

            features_for_pred = get_live_features(binance, testnet_config['symbol'], timeframes, feature_config,
                                                  feature_names)

            if features_for_pred is None:
                logging.warning(f"Feature generation failed. Sleeping...")
                time.sleep(testnet_config['polling_interval_seconds'])
                continue

            prediction = model.predict(pd.DataFrame(features_for_pred).T)[0]
            recent_signals.append(prediction)

            logging.info(f"Prediction for latest full candle. Raw Signal: {prediction}")
            logging.info(f"Recent signals buffer: {list(recent_signals)}")

            is_trade_state = (len(recent_signals) == persistence and all(s == 1 for s in recent_signals))

            if is_trade_state and not was_in_trade_state and position_amount == 0:
                logging.info("PERSISTENT ENTRY SIGNAL DETECTED! Placing LONG order...")
                binance.set_leverage(testnet_config['leverage'], testnet_config['symbol'])

                trade_size = 10
                entry_params = {'positionSide': 'LONG'}
                order = binance.create_order(testnet_config['symbol'], 'market', 'buy', trade_size, params=entry_params)
                logging.info(f"Placed market buy order: {order}")

                entry_price = float(order['price'])
                stop_loss_price = entry_price * (1 - testnet_config['stop_loss_pct'])
                take_profit_price = entry_price * (1 + testnet_config['take_profit_pct'])

                logging.info(f"Placing OCO stop-loss and take-profit orders...")

                sl_tp_params = {'positionSide': 'LONG', 'reduceOnly': True}

                sl_order = binance.create_order(
                    symbol=testnet_config['symbol'],
                    type='STOP_MARKET',
                    side='sell',
                    amount=trade_size,
                    params={'stopPrice': stop_loss_price, **sl_tp_params}
                )
                logging.info(f"Placed SL order at {stop_loss_price}: {sl_order}")

                tp_order = binance.create_order(
                    symbol=testnet_config['symbol'],
                    type='TAKE_PROFIT_MARKET',
                    side='sell',
                    amount=trade_size,
                    params={'stopPrice': take_profit_price, **sl_tp_params}
                )
                logging.info(f"Placed TP order at {take_profit_price}: {tp_order}")

            elif not is_trade_state and was_in_trade_state and position_amount > 0:
                logging.info("EXIT SIGNAL DETECTED! Closing open position...")
                logging.info("Cancelling existing SL/TP orders...")
                binance.cancel_all_orders(testnet_config['symbol'])

                close_params = {'positionSide': 'LONG', 'reduceOnly': True}
                close_order = binance.create_order(testnet_config['symbol'], 'market', 'sell', position_amount,
                                                   params=close_params)
                logging.info(f"Placed market sell order to close position: {close_order}")

            was_in_trade_state = is_trade_state

        except Exception as e:
            logging.error(f"An error occurred in the trading loop: {e}", exc_info=True)

        logging.info(f"Sleeping for {testnet_config['polling_interval_seconds']} seconds...")
        time.sleep(testnet_config['polling_interval_seconds'])