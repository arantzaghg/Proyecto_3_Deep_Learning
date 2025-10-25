           import pandas as pd
import mlflow
import numpy as np
from get_signals import signals
from indicators import get_indicators
from data_utils import split_data
from normalization import get_normal_stats  
from backtesting import backtest
from plots import plot_portfolio_value
from data_utils import get_asset_data, preprocess_data, get_target
from CNN_model import train_signals_cnn, get_params_space_cnn
from MLP_model import train_signals_mlp, get_params_space_mlp


def main():
    
    # Get Data
    ticker = "SONY"
    data = get_asset_data(ticker)
    train, test, val = split_data(data)

    train_data, stats = preprocess_data(train, ticker, alpha=0.03, stage="train", include_close=True)
    test_data, _ = preprocess_data(test, ticker, alpha=0.03, stage="test", stats=stats, include_close=True)
    val_data, _ = preprocess_data(val, ticker, alpha=0.03, stage="val", stats=stats, include_close=True)

    # preprocess without normalization prices
    train_data_np, stats_np = preprocess_data(train, ticker, alpha=0.03, stage="train", include_close=False)
    test_data_np, _ = preprocess_data(test, ticker, alpha=0.03, stage="test", stats=stats_np, include_close=False)
    val_data_np, _ = preprocess_data(val, ticker, alpha=0.03, stage="val", stats=stats_np, include_close=False)

    # Get target
    x_train, y_train = get_target(train_data)
    x_test, y_test = get_target(test_data)
    x_val, y_val = get_target(val_data)


    for model_name in models_to_check:
        print(f"\n==============================")
        print(f" Evaluating model: {model_name}")
        print(f"==============================")

        # --- Always load the latest version ---
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.tensorflow.load_model(model_uri)
        model.summary()

        # --- Predictions ---
        y_hat_train = model.predict(x_train)
        y_hat_test = model.predict(x_test)
        y_hat_val = model.predict(x_val)

        # --- Classify ---
        train_data_np["signal"] = np.argmax(y_hat_train, axis=1)
        test_data_np["signal"] = np.argmax(y_hat_test, axis=1)
        val_data_np["signal"] = np.argmax(y_hat_val, axis=1)

                # --- Backtest ---
        portfolio_train, final_cash_train, win_rate_train, _, _, _, _ = backtest(train_data_np, cash=1_000_000)
        portfolio_test, final_cash_test, win_rate_test, buy_test, sell_test, hold_test, total_trades_test = backtest(test_data_np, cash=1_000_000)
        portfolio_val, final_cash_val, win_rate_val, buy_val, sell_val, hold_val, total_trades_val = backtest(val_data_np, cash=final_cash_test)

        # --- Results ---
        print(f"\n--- RESULTS {model_name.upper()} ---")
        print(f"Train cash: ${final_cash_train:,.2f} | Win rate: {win_rate_train:.2%}")
        print(f"Test  cash: ${final_cash_test:,.2f} | Win rate: {win_rate_test:.2%}")
        print(f"Val   cash: ${final_cash_val:,.2f} | Win rate: {win_rate_val:.2%}")

        # --- Trade stats ---
        print(f"\n--- TRADE STATISTICS ---")
        print(f"Test set  → Buys: {buy_test:,} | Sells: {sell_test:,} | Holds: {hold_test:,} | Total trades: {total_trades_test:,}")
        print(f"Val set   → Buys: {buy_val:,}  | Sells: {sell_val:,}  | Holds: {hold_val:,}  | Total trades: {total_trades_val:,}")

        plot_portfolio_value(portfolio_train, title="Train")
        plot_portfolio_value(portfolio_test, title="Test")
        plot_portfolio_value(portfolio_val, title="Validation")

if __name__ == "__main__":
    main()