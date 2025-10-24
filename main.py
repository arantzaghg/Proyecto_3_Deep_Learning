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
 
    ticker = "AMZN"
    data = get_asset_data(ticker)
    train_data, test_data, val_data = split_data(data)

    train_data, stats = preprocess_data(train_data, ticker=ticker, alpha=0.016, stage="train", include_close=True)
    test_data, _ = preprocess_data(test_data, ticker=ticker, alpha=0.016, stage="test", stats=stats, include_close=True)
    val_data, _ = preprocess_data(val_data, ticker=ticker, alpha=0.016, stage="val", stats=stats, include_close=True)

    x_train, y_train = get_target(train_data)
    x_test, y_test = get_target(test_data)
    x_val, y_val = get_target(val_data)

    model_name = "Models"
    model_version = 1   
    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.tensorflow.load_model(model_uri)
    model.summary()

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_val = model.predict(x_val)

    y_pred_train_classes = np.argmax(y_pred_train, axis=1)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)
    y_pred_val_classes = np.argmax(y_pred_val, axis=1)

    test_data["signal"] = y_pred_test_classes
    train_data["signal"] = y_pred_train_classes 
    val_data["signal"] = y_pred_val_classes

    portfolio_train, final_cash_train, win_rate_train = backtest(train_data, cash=1000000)
    portfolio_test, final_cash_test, win_rate_test = backtest(test_data, cash=1000000)
    portfolio_val, final_cash_val, win_rate_val = backtest(val_data, cash=final_cash_test)

    print(f"\n--- RESULTS TRAIN: ---")
    print(f"Final cash: ${final_cash_train:,.2f}")
    print(f"Win rate: {win_rate_train:.2%}")

    print(f"\n--- RESULTS TEST: ---")
    print(f"Final cash: ${final_cash_test:,.2f}")
    print(f"Win rate: {win_rate_test:.2%}")

    print(f"\n--- RESULTS VALIDATION: ---")
    print(f"Final cash: ${final_cash_val:,.2f}")
    print(f"Win rate: {win_rate_val:.2%}")

    plot_portfolio_value(portfolio_train, title="Train")
    plot_portfolio_value(portfolio_test, title="Test")
    plot_portfolio_value(portfolio_val, title="Validation")

if __name__ == "__main__":
    main()