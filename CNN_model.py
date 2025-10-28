import tensorflow as tf
import mlflow
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def build_model(params, input_shape):

    """
    Builds and compiles a CNN model based on the provided parameters.
    
    Parameters:
    params (dict): Dictionary containing model hyperparameters.
    input_shape (int): The shape of the input data.
    
    Returns:
    tf.keras.Model: Compiled CNN model.
    """

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape, 1)))

    num_filters = params.get("conv_filters", 32)
    conv_layers = params.get("conv_layers", 2)
    activation  = params.get("activation", "relu")

    for _ in range(conv_layers):
        model.add(tf.keras.layers.Conv1D(num_filters, kernel_size=5, padding="causal", activation=activation))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        num_filters *= 2

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params.get("dense_units", 64), activation=activation))
    model.add(tf.keras.layers.Dense(3, activation="softmax")) 

    model.compile(optimizer=params.get("optimizer", "adam"),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def get_params_space_cnn():

    """
    Defines a list of hyperparameter combinations for CNN model training.
    
    Returns:
    list[dict]: List of dictionaries containing different hyperparameter settings.
    """

    return [
        {"conv_layers": 3, "conv_filters": 64,  "activation": "sigmoid", "dense_units": 96},  
        {"conv_layers": 3, "conv_filters": 96,  "activation": "sigmoid", "dense_units": 128},   
        {"conv_layers": 3, "conv_filters": 64,  "activation": "relu", "dense_units": 128}, 
        {"conv_layers": 3, "conv_filters": 64, "activation": "sigmoid", "dense_units": 128}, 
    ]


def train_signals_cnn(X_train, y_train, X_test, y_test, X_val, y_val, params_cnn, epochs=10, batch_size=32):

    """
    Trains multiple CNN models with different hyperparameters and logs results to MLflow.

    Parameters:
    X_train (np.ndarray): Training feature set.
    y_train (np.ndarray): Training labels.
    X_test (np.ndarray): Testing feature set.
    y_test (np.ndarray): Testing labels.
    X_val (np.ndarray): Validation feature set.
    y_val (np.ndarray): Validation labels.
    params_cnn (list[dict]): List of hyperparameter combinations for CNN models.
    epochs (int): Number of training epochs.
    batch_size (int): Size of training batches.

    Returns:
    None
    """

    input_shape = X_train.shape[1]  

    print("Training models...")
    for params in params_cnn:
        run_name = (
            f"conv{params['conv_layers']}_filters{params['conv_filters']}"
            f"_dense{params['dense_units']}_activation{params['activation']}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("CNN", run_name)
            model = build_model(params, input_shape)
            hist = model.fit(
                X_train, y_train,
                epochs=epochs,
                validation_data=(X_test, y_test),
                batch_size=batch_size,
                verbose=2
            )

            y_pred_probs_test = model.predict(X_test)
            y_pred_test = np.argmax(y_pred_probs_test, axis=1)

            y_pred_probs_val = model.predict(X_val)
            y_pred_val = np.argmax(y_pred_probs_val, axis=1)

            f1_test = f1_score(y_test, y_pred_test, average="weighted")
            test_accuracy = accuracy_score(y_test, y_pred_test)

            f1_val = f1_score(y_val, y_pred_val, average="weighted")

            
            mlflow.log_metrics({
                "val_f1_score": f1_test,
                "test_accuracy": float(test_accuracy),
                "test_f1_score": float(f1_val),
            })

            final_metrics = {
                "val_accuracy": float(hist.history["val_accuracy"][-1]),
                "val_loss": float(hist.history["val_loss"][-1]),
            }
            print(f"[{run_name}] Final metrics: {final_metrics}")