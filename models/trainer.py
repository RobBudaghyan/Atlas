import pandas as pd
import lightgbm as lgb
import joblib
import os
import logging
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(symbol, features_dir, model_dir, validation_config):
    """
    Trains a LightGBM model using walk-forward validation.
    """
    logging.info("Starting model training...")

    pycache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
    if os.path.exists(pycache_dir):
        logging.info(f"Clearing cache directory: {pycache_dir}")
        shutil.rmtree(pycache_dir)

    features_path = os.path.join(features_dir, f"{symbol.replace('/', '_')}_features.csv")
    if not os.path.exists(features_path):
        logging.error("Feature file not found. Please generate features first.")
        return

    df = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)

    features = [col for col in df.columns if 'target' not in col and 'close_time' not in col]
    X = df[features]
    y = df['target']

    train_size = validation_config['train_size']
    test_size = validation_config['test_size']

    num_folds = int(np.floor((len(df) - train_size) / test_size))
    if num_folds <= 0:
        logging.error("Not enough data for even one walk-forward fold.")
        return

    logging.info(f"Starting walk-forward validation with {num_folds} folds.")

    for fold in tqdm(range(num_folds), desc="Walk-Forward Training"):
        start_index = fold * test_size
        train_end = start_index + train_size
        test_end = train_end + test_size

        if test_end > len(df):
            break

        X_train, y_train = X.iloc[start_index:train_end], y.iloc[start_index:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]


        model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='accuracy',
            callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)]
        )

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        logging.info(f"Fold {fold + 1}/{num_folds} Test Accuracy: {accuracy:.4f}")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{symbol.replace('/', '_')}_model_fold_{fold + 1}.joblib")
        joblib.dump(model, model_path)

    logging.info(f"Finished training. Models saved in {model_dir}")


if __name__ == '__main__':
    from config import SYMBOL, FEATURES_DIRECTORY, MODEL_DIRECTORY, WALK_FORWARD_VALIDATION

    train_model(SYMBOL, FEATURES_DIRECTORY, MODEL_DIRECTORY, WALK_FORWARD_VALIDATION)