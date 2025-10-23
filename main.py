import pandas as pd
import mlflow
from get_signals import signals
from indicators import get_indicators
from data_utils import split_data
from normalization import get_normal_stats  
from backtesting import backtest
from plots import plot_portfolio_value_train
from data_utils import get_asset_data, preprocess_data, get_target
from CNN_model import train_signals_cnn, get_params_space_cnn
from MLP_model import train_signals_mlp, get_params_space_mlp


def main():
 
    ticker = "AMZN"
    data = get_asset_data(ticker)
    train_data, test_data, val_data = split_data(data)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    val_data = preprocess_data(val_data)

    x_train, y_train = get_target(train_data)
    x_test, y_test = get_target(test_data)
    x_val, y_val = get_target(val_data)

    model_type = "CNN"
    model_version = "latest"
    model = mlflow.tensorflow.load_model(f"models:/{model_type}/{model_version}")

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
    portfolio_val, final_cash_val, win_rate_val = backtest(val_data, cash=1000000)

    for name, portfolio, cash, win_rate in [
        ("TRAIN", portfolio_train, final_cash_train, win_rate_train),
        ("TEST", portfolio_test, final_cash_test, win_rate_test),
        ("VALIDATION", portfolio_val, final_cash_val, win_rate_val),
    ]:
        print(f"\n--- RESULTS: {name} ---")
        print(f"Final cash: ${cash:,.2f}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Initial cash: ${1000000:,.2f}")
        total_return = (cash / 1000000 - 1) * 100
        print(f"Total return: {1000000:.2f}%")




    
if __name__ == "__main__":
    main()