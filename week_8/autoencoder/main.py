import numpy as np
import matplotlib.pyplot as plt
import copy

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam


(x_train, _), (x_test, _) = mnist.load_data(path="mnist.npz")

x_train_noised = copy.copy(x_train)
x_test_noised = copy.copy(x_test)


def add_noise(x):
    mean = 0
    sigma = 3
    noise = np.random.normal(mean, sigma, x[0].shape)
    for i in range(len(x)):
        x[i] = x[i] + noise
    return x


x_train_noised = add_noise(x_train_noised)
x_test_noised = add_noise(x_test_noised)


def normal_and_reshape(x):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    x = x.reshape(x.shape[0], 28, 28, 1)

    return x


x_train = normal_and_reshape(x_train)
x_test = normal_and_reshape(x_test)

x_train_noised = normal_and_reshape(x_train_noised)
x_test_noised = normal_and_reshape(x_test_noised)

input_shape = x_train.shape[1:]


def train():
    model = Sequential([

        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),

        Conv2DTranspose(32, (3, 3), activation='relu'),
        Conv2DTranspose(64, (3, 3), activation='relu'),
        Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    model.summary()

    opt = Adam(learning_rate=0.0001)

    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.fit(x_train_noised, x_train, epochs=12, batch_size=150)
    model.save('model.h5')


def show_results(noisy, pure, number_of_visualizations):

    noisy = noisy[:number_of_visualizations]
    pure = pure[:number_of_visualizations]
    model = load_model("model.h5")
    denoised = model.predict(noisy)

    for i in range(0, number_of_visualizations):
        noisy_img = noisy[i][:, :, 0]
        pure_img = pure[i][:, :, 0]
        denoised_img = denoised[i][:, :, 0]

        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(8, 3.5)
        # Plot sample and reconstruciton
        axes[0].imshow(noisy_img)
        axes[0].set_title('Noisy image')
        axes[1].imshow(pure_img)
        axes[1].set_title('Pure image')
        axes[2].imshow(denoised_img)
        axes[2].set_title('Denoised image')

        plt.show()


# train()
show_results(noisy=x_train_noised, pure=x_train, number_of_visualizations=10)


