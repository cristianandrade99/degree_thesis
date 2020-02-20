import tensorflow as tf
from tensorflow.keras import layers

class Sample(layers.Layer):
    def call(self,mean,logvar):
        return tf.random.normal(tf.shape(mean))*tf.exp(0.5*logvar)+mean

# Class for the layer that applies tanh
class Tanh(layers.Layer):
    def call(self, inputs):
        return tf.keras.activations.tanh(inputs)

# Class for the layer that applies sigmoid
class Sigmoid(layers.Layer):
    def call(self, inputs):
        return tf.keras.activations.sigmoid(inputs)
