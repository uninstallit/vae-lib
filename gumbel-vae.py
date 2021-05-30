import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# source: https://github.com/EderSantana/gumbel

batch_size = 10
data_dim = 784

M = 10  # classes
N = 30  # how many distributions

nb_epoch = 100
epsilon_std = 0.01
anneal_rate = 0.0003
min_temperature = 0.5

tau = tf.Variable(5.0, dtype=tf.float32)


class Sampling(keras.layers.Layer):
    def call(self, logits_y):
        u = tf.random.uniform(tf.shape(logits_y), 0, 1)
        y = logits_y - tf.math.log(
            -tf.math.log(u + 1e-20) + 1e-20
        )  # logits + gumbel noise
        y = tf.nn.softmax(tf.reshape(y, (-1, N, M)) / tau)
        y = tf.reshape(y, (-1, N * M))
        return y


encoder_inputs = keras.Input(shape=(data_dim,))
x = keras.layers.Dense(512, activation="relu")(encoder_inputs)
x = keras.layers.Dense(256, activation="relu")(x)
logits_y = keras.layers.Dense(M * N, name="logits_y")(x)
z = Sampling()(logits_y)
encoder = keras.Model(encoder_inputs, [logits_y, z], name="encoder")
encoder.build(encoder_inputs)

print(encoder.summary())

decoder_inputs = keras.Input(shape=(N * M,))
x = keras.layers.Dense(256, activation="relu")(decoder_inputs)
x = keras.layers.Dense(512, activation="relu")(x)
decoder_outputs = keras.layers.Dense(data_dim, activation="sigmoid")(x)
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.build(decoder_inputs)

print(decoder.summary())


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x):
        _, z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    @tf.function
    def train_step(self, data):
        x = data

        with tf.GradientTape(persistent=True) as tape:
            logits_y, z = self.encoder(x, training=True)
            x_hat = self.decoder(z, training=True)

            q_y = tf.reshape(logits_y, (-1, N, M))
            q_y = tf.nn.softmax(q_y)
            log_q_y = tf.math.log(q_y + 1e-20)
            kl = q_y * (log_q_y - tf.math.log(1.0 / M))
            kl = tf.math.reduce_sum(kl, axis=(1, 2))
            elbo = data_dim * self.bce(x, x_hat) - kl

        grads = tape.gradient(elbo, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(elbo)
        return {"loss": self.loss_tracker.result()}


def main():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae = VAE(encoder, decoder, name="vae-model")
    vae_inputs = (None, data_dim)
    vae.build(vae_inputs)
    vae.compile(optimizer="adam")
    vae.fit(x_train, shuffle=True, epochs=1, batch_size=batch_size)

    # vae.fit(x_train, shuffle=True, epochs=1, batch_size=batch_size)

    for e in range(nb_epoch):
        vae.fit(x_train, shuffle=True, epochs=1, batch_size=batch_size)
        keras.backend.set_value(tau, np.max([keras.backend.get_value(tau) * np.exp(-anneal_rate * e), min_temperature]),)

    # vis
    # argmax_y = keras.backend.max(tf.reshape(logits_y, (-1, N, M)), axis=-1, keepdims=True)
    # argmax_y = keras.backend.equal(tf.reshape(logits_y, (-1, N, M)), argmax_y)
    # encoder = keras.backend.function([x], [argmax_y, x_hat])

    x = np.reshape(x_test[0], (1, 28 * 28))
    logits_y, z = vae.encoder([x], training=False)
    x_hat = vae.decoder.predict(z)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    axs[0].imshow(x.reshape(28, 28), cmap='gray') 
    axs[1].imshow(z.numpy().reshape(N, M), cmap='gray')
    axs[2].imshow(x_hat.reshape(28, 28), cmap='gray')
    fig.suptitle('Gumbel-Softmax Variational Autoencoder')
    plt.show()

#     from sklearn.manifold.t_sne import TSNE
#     tsne = TSNE(metric='hamming')
#     viz = tsne.fit_transform(C)

# from agnez import embedding2dplot
# _ = embedding2dplot(viz, y_test[:6000], show_median=False)


if __name__ == "__main__":
    main()
