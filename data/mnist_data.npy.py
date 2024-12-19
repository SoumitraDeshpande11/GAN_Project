import numpy as np
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

# Save the preprocessed dataset
np.save('../data/mnist_data.npy', x_train)
print("MNIST data saved to 'data/mnist_data.npy'.")
