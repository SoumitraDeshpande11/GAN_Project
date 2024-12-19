import matplotlib.pyplot as plt
import numpy as np

def sample_images(generator, epoch, n=5, save_dir="../outputs/samples/"):
    noise = np.random.normal(0, 1, (n * n, 100))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(n, n)
    count = 0
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(gen_images[count, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
            count += 1
    plt.savefig(f"{save_dir}epoch_{epoch}.png")
    plt.close()
