
import tensorflow as tf
import mlflow

def build_model(params, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape, 1)))

    num_filters = params.get("conv_filters", 32)
    conv_layers = params.get("conv_layers", 2)
    activation  = params.get("activation", "relu")

    for _ in range(conv_layers):
        model.add(tf.keras.layers.Conv1D(num_filters, kernel_size=3, padding="same", activation=activation))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        num_filters *= 2

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params.get("dense_units", 64), activation=activation))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))  # señal compra (1) / no compra (0)

    model.compile(optimizer=params.get("optimizer", "adam"),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def get_params_space_cnn():
    return [
        {"conv_layers": 2, "conv_filters": 32, "activation": "relu",    "dense_units": 64},
        {"conv_layers": 3, "conv_filters": 64, "activation": "relu",    "dense_units": 32},
        {"conv_layers": 2, "conv_filters": 32, "activation": "sigmoid", "dense_units": 64},
    ]

# ------- Entrenamiento (usa MLflow ya configurado) -------
def train_signals_cnn(X_train, y_train, X_test, y_test, params_cnn, epochs=10, batch_size=32):

    input_shape = X_train.shape[1]  # (window, n_features), si no funciona quitar :

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
            final_metrics = {
                "val_accuracy": float(hist.history["val_accuracy"][-1]),
                "val_loss": float(hist.history["val_loss"][-1]),
            }
            print(f"[{run_name}] Final metrics: {final_metrics}")
