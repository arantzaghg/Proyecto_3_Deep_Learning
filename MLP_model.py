import tensorflow as tf
import mlflow

def build_model(params, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    
    activation = params.get("activation", "relu")
    dense_layers = params.get("dense_layers", 2)
    dense_units = params.get("dense_units", 64)
    
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(dense_units, activation=activation))

    model.add(tf.keras.layers.Dense(3, activation="softmax"))  # señal compra (1) / no compra (0)

    model.compile(optimizer=params.get("optimizer", "adam"),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model


def get_params_space_mlp():
    return [
        {"dense_layers": 2, "dense_units": 64, "activation": "relu"},
        {"dense_layers": 3, "dense_units": 32, "activation": "relu"},
        {"dense_layers": 2, "dense_units": 64, "activation": "sigmoid"},
    ]


def train_signals_mlp(X_train, y_train, X_test, y_test, params_mlp, epochs=10, batch_size=32):

    input_shape = X_train.shape[1]

    print("Training models...")
    for params in params_mlp:
        run_name = (
            f"dense{params['dense_layers']}_units{params['dense_units']}"
            f"_activation{params['activation']}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("MPL", run_name)
            model = build_model(params, input_shape)
            hist = model.fit(
                X_train, y_train,
                epochs=epochs,
                validation_data=(X_test, y_test),
                batch_size=batch_size,
                verbose=2
            )
            final_metrics = {
                "val_accuracy": float(hist.history["val_accuracy"][-1]),
                "val_loss": float(hist.history["val_loss"][-1]),
            }
            print(f"[{run_name}] Final metrics: {final_metrics}")


