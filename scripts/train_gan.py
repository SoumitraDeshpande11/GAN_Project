import numpy as np
from tensorflow.keras.optimizers import Adam
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils import train_gan

# Load preprocessed data
x_train = np.load('../data/mnist_data.npy')

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])

# Freeze discriminator for GAN
discriminator.trainable = False

# Build and compile GAN
from tensorflow.keras import Model, Input
gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")

# Train the GAN
train_gan(generator, discriminator, gan, x_train, epochs=5000, batch_size=64)
