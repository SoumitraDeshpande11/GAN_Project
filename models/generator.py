from tensorflow.keras import layers, Sequential

def build_generator():
    model = Sequential([
        layers.Dense(256, activation="relu", input_dim=100),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1024, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(28 * 28 * 1, activation="tanh"),
        layers.Reshape((28, 28, 1))
    ])
    return model
