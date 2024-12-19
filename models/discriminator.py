from tensorflow.keras import layers, Sequential

def build_discriminator():
    model = Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model
