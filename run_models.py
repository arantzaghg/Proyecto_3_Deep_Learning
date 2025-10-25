from data_utils import get_asset_data, split_data, preprocess_data, get_target
from CNN_model import train_signals_cnn, get_params_space_cnn
from MLP_model import train_signals_mlp, get_params_space_mlp
import mlflow
import mlflow.tensorflow

def run_models():

    # Load Data
    ticker = "SONY"
    data = get_asset_data(ticker)
    train_data, test_data, val_data = split_data(data)

    # Generate Indicators and Signals
    train_data, stats = preprocess_data(train_data, ticker, alpha=0.010, stage="train", include_close=True)
    test_data, _ = preprocess_data(test_data, ticker, alpha=0.010, stage="test", stats=stats, include_close=True)

    # Get target
    x_train, y_train = get_target(train_data)
    x_test, y_test = get_target(test_data)

    params_cnn = get_params_space_cnn()
    params_mlp = get_params_space_mlp()


    mlflow.tensorflow.autolog()

    # Train CNN models
    mlflow.set_experiment("CNN model")
    train_signals_cnn(x_train, y_train, x_test, y_test, params_cnn, epochs=50, batch_size=32)

    # Train MLP models
    mlflow.set_experiment("MLP model")
    train_signals_mlp(x_train, y_train, x_test, y_test, params_mlp, epochs=50, batch_size=32)

if __name__ == "__main__":
    run_models()

