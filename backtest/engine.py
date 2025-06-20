import pandas as pd
import numpy as np
import joblib
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_backtest(symbol, features_dir, model_dir, backtest_config):
    """
    Performs a backtest with Signal Filtering and a Cooldown period.
    """
    logging.info("Running backtest with Signal Filtering and Cooldown...")
    features_path = os.path.join(features_dir, f"{symbol.replace('/', '_')}_features.csv")
    df = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)

    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.joblib')])
    if not model_files:
        logging.error("No trained models found.")
        return

    model = joblib.load(os.path.join(model_dir, model_files[-1]))
    feature_names = model.feature_name_

    logging.info("Generating all predictions in a single batch...")
    df['signal'] = model.predict(df[feature_names])
    logging.info("Predictions generated.")

    persistence = backtest_config.get('signal_persistence', 3)
    df['persistent_signal'] = df['signal'].rolling(window=persistence).sum()
    df['entry_signal'] = (df['persistent_signal'] == persistence) & (df['persistent_signal'].shift(1) < persistence)

    logging.info("Simulating trades with advanced logic...")
    capital = backtest_config['initial_capital']
    position_size = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    equity_curve = [capital]
    trades = []

    cooldown_period = pd.to_timedelta(backtest_config.get('cooldown_minutes', 15), 'm')
    last_trade_exit_time = pd.Timestamp.min

    for i in tqdm(range(1, len(df)), desc="Backtest Simulation"):
        current_time = df.index[i]
        current_price = df['close_1m'].iloc[i]
        high_price = df['high_1m'].iloc[i]
        low_price = df['low_1m'].iloc[i]

        if position_size > 0:
            if low_price <= stop_loss_price:
                exit_price = stop_loss_price
                capital += position_size * exit_price * (1 - backtest_config['fees'])
                trades[-1].update({'exit_time': current_time, 'exit_price': exit_price, 'exit_reason': 'SL'})
                position_size = 0
                last_trade_exit_time = current_time
            elif high_price >= take_profit_price:
                exit_price = take_profit_price
                capital += position_size * exit_price * (1 - backtest_config['fees'])
                trades[-1].update({'exit_time': current_time, 'exit_price': exit_price, 'exit_reason': 'TP'})
                position_size = 0
                last_trade_exit_time = current_time

        if current_time < last_trade_exit_time + cooldown_period:
            equity_curve.append(capital + (position_size * current_price))
            continue

        if position_size == 0 and df['entry_signal'].iloc[i]:
            entry_price = current_price * (1 + backtest_config['slippage'])

            stop_loss_price = entry_price * (1 - backtest_config.get('stop_loss_pct', 0.02))
            take_profit_price = entry_price * (1 + backtest_config.get('take_profit_pct', 0.04))

            trade_capital = capital * backtest_config['position_sizing']
            position_size = trade_capital / entry_price
            capital -= position_size * entry_price * (1 + backtest_config['fees'])
            trades.append(
                {'entry_time': current_time, 'entry_price': entry_price, 'type': 'long', 'size': position_size})

        elif position_size > 0 and not (df['persistent_signal'].iloc[i] == persistence):
            exit_price = current_price * (1 - backtest_config['slippage'])
            capital += position_size * exit_price * (1 - backtest_config['fees'])
            trades[-1].update({'exit_time': current_time, 'exit_price': exit_price, 'exit_reason': 'Signal'})
            position_size = 0
            last_trade_exit_time = current_time

        equity_curve.append(capital + (position_size * current_price))

    logging.info("Calculating performance metrics...")
    equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    daily_returns = equity_series.resample('D').last().pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0

    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() * 100

    trade_log = pd.DataFrame(trades)
    completed_trades = pd.DataFrame()
    if 'exit_price' in trade_log.columns and not trade_log.empty:
        completed_trades = trade_log.dropna(subset=['exit_price']).copy()
        if not completed_trades.empty:
            completed_trades.loc[:, 'return'] = (completed_trades['exit_price'] - completed_trades['entry_price']) / \
                                                completed_trades['entry_price']
            win_rate = (completed_trades['return'] > 0).mean() * 100
            avg_trade_return = completed_trades['return'].mean() * 100
            completed_trades.loc[:, 'holding_time'] = completed_trades['exit_time'] - completed_trades['entry_time']
            avg_holding_time = completed_trades['holding_time'].mean()
        else:
            win_rate, avg_trade_return, avg_holding_time = 0, 0, pd.Timedelta(0)
    else:
        win_rate, avg_trade_return, avg_holding_time = 0, 0, pd.Timedelta(0)

    logging.info("--- Backtest Results ---")
    logging.info(f"Total Return: {total_return:.2f}%")
    logging.info(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
    logging.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logging.info(f"Win Rate: {win_rate:.2f}%")
    logging.info(f"Average Trade Return: {avg_trade_return:.3f}%")
    logging.info(f"Average Holding Time: {avg_holding_time}")
    logging.info(f"Total Trades: {len(completed_trades)}")


    plt.figure(figsize=(14, 7))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.plot(equity_series.index, equity_series.values, label="Equity", linewidth=2.5, color="#007acc")

    plt.fill_between(equity_series.index, equity_series.values, color="#007acc", alpha=0.2)

    plt.title(f"{symbol} Backtest Equity Curve", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig("equity_curve.png", dpi=300)
    plt.close()
    logging.info("Equity curve plot saved to equity_curve.png")


    trade_log.to_csv("backtest_trades.csv", index=False)
    logging.info("Trade log saved to backtest_trades.csv")


if __name__ == '__main__':
    from config import SYMBOL, FEATURES_DIRECTORY, MODEL_DIRECTORY, BACKTESTING_CONFIG

    BACKTESTING_CONFIG['stop_loss_pct'] = 0.02
    BACKTESTING_CONFIG['take_profit_pct'] = 0.07
    BACKTESTING_CONFIG['signal_persistence'] = 3
    BACKTESTING_CONFIG['cooldown_minutes'] = 10
    run_backtest(SYMBOL, FEATURES_DIRECTORY, MODEL_DIRECTORY, BACKTESTING_CONFIG)