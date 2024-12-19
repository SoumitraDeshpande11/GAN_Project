from tensorflow.keras.models import save_model

def save_models(generator, discriminator, epoch, save_dir="../checkpoints/"):
    generator.save(f"{save_dir}generator_epoch_{epoch}.h5")
    discriminator.save(f"{save_dir}discriminator_epoch_{epoch}.h5")

def train_gan(generator, discriminator, gan, data, epochs=5000, batch_size=64, sample_interval=100):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_images = data[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)

        # Print progress
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
            sample_images(generator, epoch)
            save_models(generator, discriminator, epoch)
