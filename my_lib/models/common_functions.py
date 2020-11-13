import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras import backend as K


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




# returns 2 outputs (sample, log_denisty)
def sampling2(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    
    z_new = z_mean + K.exp(0.5 * z_log_var) * epsilon
    q_z = log_normal_pdf(z_new, z_mean, z_log_var)
    return z_new, q_z




class Linear(layers.Layer):
    def __init__(self, input_dim=32, output_dim=32, **kwargs):
        
        super().__init__(**kwargs)
        
        self.w = self.add_weight(
            shape=(input_dim, output_dim), initializer="random_normal", trainable=True, name="w",
        )
        self.b = self.add_weight(shape=(output_dim,), initializer="random_normal", trainable=True, name="b")


    def call(self, inputs):
        raw = tf.matmul(inputs, self.w) + self.b
        return raw
#         return tf.nn.softmax(raw)
    
    
   
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)
    
