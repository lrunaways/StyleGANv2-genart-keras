import numpy as np
import tensorflow as tf
from typing import Any, List, Tuple, Union
from tensorflow.keras.layers import Layer, Lambda, Add, Conv2DTranspose
from upfirdn_2d import upsample_2d, downsample_2d


def float_img_to_int8(images):
    lo, hi = -1, 1
    # scale = 255 / (-1 - 1)
    # images = img * scale + (0.5 - 1 * scale)
    # return tf.saturate_cast(images, tf.uint8).numpy()
    img_list = []
    for image in images:
        image = (image - lo) * (255 / (hi - lo))
        img_list.append(np.rint(image).clip(0, 255).astype(np.uint8))
    return img_list


def normalize_2nd_moment(x, axis=-1, eps=1e-8):
    return Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.math.square(x), axis=axis, keepdims=True) + eps),
                  name='normalize_2nd_moment')(x)


def lerp(latents, dlatent_avg, psi, truncation_cutoff=None):
    # if truncation_cutoff is None:
    #     layer_psi = truncation_psi
    # else:
    #     layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
    return dlatent_avg + (latents - dlatent_avg) * psi

# Upsampling block.
def upsample(y, resample_kernel, data_format, res_name):
    y = Lambda(lambda x: upsample_2d(x, k=resample_kernel, data_format=data_format), name=res_name+'/up')(y)
    return y


class Dense(Layer):
  def __init__(self, units=32, name="Dense", constant_b=0, lrmul=1.0, act='lrelu'):
      super(Dense, self).__init__(name=name)
      self.units = units
      # self._name = name
      self.constant_b = constant_b
      self.act = act
      self.lrmul = lrmul


  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True, name='weight')
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True, name='bias')

  def call(self, inputs):  # Defines the computation from inputs to outputs
      fan_in = tf.math.reduce_prod(self.w.shape[:-1])
      he_std = 1.0 / tf.math.sqrt(tf.dtypes.cast(fan_in, tf.float32))
      runtime_coef = he_std * self.lrmul
      # runtime_coef = 1.0
      # self.lrmul = 1.0

      basic_out = tf.matmul(inputs, self.w*runtime_coef) + self.b*self.lrmul - self.constant_b
      if self.act == 'lrelu':
          out = tf.nn.leaky_relu(basic_out, alpha=0.2)*tf.math.sqrt(2.0)
      elif self.act == 'linear' or self.act is None:
          out = basic_out
      else:
          raise ValueError('Activation is unsupported.')
      return out


class Dlatent_avg(Layer):
  def __init__(self, units=512, name='dlatent_avg'):
      super(Dlatent_avg, self).__init__(name=name)
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(1, self.units),dtype='float32'), trainable=True, name='latent')

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return self.w + tf.reduce_mean(inputs*0.0)

