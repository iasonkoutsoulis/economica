# Author S. Iason Koutsoulis, March 2023.
# This code aims at generating a representative economic agent who will live for T=70 and pass, 
# while endowing his offspring with his knowledge, which is twofold: 
# 1) The state of the world in his last t, X_t and
# 2) his action set, Y_t.

# I'm thinking I start with a simple Autoencoder NN in a less-than-extreme-regimes state-of-the-world environment. 
# Ideally, the reduction from multiple inputs to a simplified hidden layer takes into account some information theory.
# The agent will have T years to live, of which 'k' will be in one state and T-k will be in the other state.

# Later on, we shall move on with a continuum of states, only bounded within extreme bounds (all-is-well and we-die-now).
# I will have to read up on the literature to enable this smooth transition.


# Let us start with an encoder toy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.imshow(x_train[0], cmap="gray")
# plt.show()

x_train[0].shape
x_train[0]
x_train = x_train/255.0
x_test = x_test/255.0

encoder_input = keras.Input(shape=(28,28,1), name = "img") # is no. of channels
# make a visualisable state of the economy! then you can do nice NNs

x = keras.layers.Flatten()(encoder_input) #instead of the sequential

# this is the reduction of the encoder
encoder_output = keras.layers.Dense(64, activation="relu")(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")

# take in the input
decoder_input = keras.layers.Dense(784, activation="relu")(encoder_output) # this )( is a *

# push out the output
decoder_output = keras.layers.Reshape((28, 28, 1))(decoder_input)

opt = keras.optimizers.Adam(learning_rate=0.001)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
autoencoder.compile(opt, loss="mse")

autoencoder.fit(x_train, x_train, epochs=3, batch_size=32, validation_split=0.1) 
#mapping features to features to get a reduced form of the problem

example = encoder.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]
# print(example)
# example.shape

# plt.imshow(example.reshape(8,8), cmap="gray")
# plt.show()

plt.imshow(x_test[0], cmap="gray")
plt.show()

ae_out = autoencoder.predict([x_test[0].reshape(-1, 28, 28, 1)])[0]
plt.imshow(ae_out, cmap="gray")
plt.show()

# So, the point of rational destruction of information is very much at 
# play here, which makes me especially happy, because we can now model
# this mathematically - or rather, rationalize the parameters that make
# the agent perceive reality as a function of what it is.






