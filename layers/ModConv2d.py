import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export

from upfirdn_2d import *
from layers.other import Dense, normalize_2nd_moment

# ToRGB block.
def torgb(x, y, latents, res_name, is_grouped, style_strength_map=None): # res = 2..resolution_log2
    if not is_grouped:
        t = ModConv2d(rank=2, sampling=None, filters=3, kernel_size=1, demodulate=False, noise=True, act=None, name=res_name+'/ToRGB')([x, latents[0:1, -1]])
    else:
        t = ModConv2d_grouped(rank=2, sampling=None, filters=3, kernel_size=1, demodulate=False, noise=True, act=None, name=res_name+'/ToRGB')([x, latents])
        t = tf.reduce_sum(t * style_strength_map, axis=1)
    if y is not None:
        t += tf.cast(y, t.dtype)
    return t

class ModConv2d(Layer):
  """Abstract N-D convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
  """

  def __init__(self, rank,
               filters,
               kernel_size,
               sampling, # [None, 'up', 'down']
               strides=1,
               act='lrelu',
               noise=True,
               demodulate=True,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(ModConv2d, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.noise = noise
    self.demodulate = demodulate
    self.act = act
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = [InputSpec(ndim=self.rank + 2), InputSpec(ndim=self.rank)]
    self.sampling = sampling

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape[0])
    input_channel = self._get_input_channel(input_shape)
    kernel_shape = self.kernel_size + (input_channel, self.filters)
    self.modulate_style = Dense(units=input_shape[-1], constant_b=0.0, act=None, name='mod_weight')
    self.noise_strength = self.add_weight(
        name='noise_strength',
        shape=1,
        initializer=tf.initializers.zeros(),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=False,
        dtype=self.dtype)
    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
      conv_inputs = inputs[0]
      print('styled: ', conv_inputs)
      style = inputs[1]

      weights = self.kernel
      he_std = 1.0 / tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(weights.shape[:-1]), tf.float32))
      runtime_coef = he_std * 1.0
      # runtime_coef = 1.0
      weights = weights*runtime_coef

      style = self.modulate_style(style) + 1.0

      if self.demodulate:
        style *= 1 / tf.reduce_max(tf.abs(style))  # Pre-normalize to avoid float16 overflow.

      weights = weights*style[0, np.newaxis, np.newaxis, :, np.newaxis]

      # Demodulate
      if self.demodulate: ##########??????
          d = tf.math.rsqrt(tf.math.reduce_sum(tf.math.square(weights), axis=[0, 1, 2]) + 1e-8)  # [BO] Scaling factor.
          weights *= d[np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO] Scale output feature maps.

      # conv_inputs = conv_inputs*style[0, np.newaxis, np.newaxis, :]  # ##################

      # Convolve
      padding = 0
      kernel = self.kernel_size[0]
      resample_kernel = [1,3,3,1]
      data_format = 'NHWC' #'NCHW'
      if self.sampling == 'up':
          x = upsample_conv_2d(conv_inputs, weights, data_format=data_format, k=resample_kernel, padding=padding)
      elif self.sampling == 'down':
          x = conv_downsample_2d(conv_inputs, weights, data_format=data_format, k=resample_kernel, padding=padding)
      else:
          padding_mode = {0: 'SAME', -(kernel // 2): 'VALID'}[padding]
          x = tf.nn.conv2d(conv_inputs, weights, data_format=data_format, strides=[1, 1, 1, 1], padding=padding_mode)

      ##############################
      if self.noise:
          noise = tf.random.normal([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1], dtype=x.dtype)
          x += noise*self.noise_strength/2

      x = nn.bias_add(x, self.bias, data_format=data_format)
      if self.act == 'lrelu':
        x = tf.nn.leaky_relu(x, alpha=0.2)*tf.math.sqrt(2.0)
      elif self.act == 'linear' or self.act is None:
        pass
      else:
        raise ValueError('Activation is unsupported.')
      return x

  def compute_output_shape(self, input_shape):
    input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(ModConv2d, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding


class ModConv2d_grouped(Layer):
  """Abstract N-D convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
  """

  def __init__(self, rank,
               filters,
               kernel_size,
               sampling, # [None, 'up', 'down']
               strides=1,
               act='lrelu',
               noise=True,
               demodulate=True,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(ModConv2d_grouped, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.noise = noise
    self.demodulate = demodulate
    self.act = act
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = [InputSpec(ndim=self.rank + 2), InputSpec(ndim=self.rank + 1)]
    self.sampling = sampling

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape[0])
    input_channel = self._get_input_channel(input_shape)
    kernel_shape = self.kernel_size + (input_channel, self.filters)
    self.modulate_style = Dense(units=input_shape[-1], constant_b=0.0, act=None, name='mod_weight')
    self.noise_strength = self.add_weight(
        name='noise_strength',
        shape=1,
        initializer=tf.initializers.zeros(),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=False,
        dtype=self.dtype)
    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
      conv_inputs = inputs[0]
      style = inputs[1][0]

      weights = self.kernel[np.newaxis]
      # print(f"conv_inputs: {conv_inputs}, style: {style}, weights: {weights}")
      he_std = 1.0 / tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(weights.shape[:-1]), tf.float32))
      runtime_coef = he_std * 1.0
      weights = weights*runtime_coef

      # Modulate.
      style = self.modulate_style(style) + 1.0
      if self.demodulate: #################################
        style *= 1 / tf.reduce_max(tf.abs(style), axis=1, keepdims=True)  # Pre-normalize to avoid float16 overflow.
      weights = weights*style[:, np.newaxis, np.newaxis, :, np.newaxis]


      # print('demod')
      # Demodulate
      if self.demodulate:############
          d = tf.math.rsqrt(tf.math.reduce_sum(tf.math.square(weights), axis=[1, 2, 3], keepdims=True) + 1e-8)  # [BO] Scaling factor.
          weights *= d # [BkkIO] Scale output feature maps.

      # print("conv_inputs before reshaping", conv_inputs)
      # conv_inputs = tf.reshape(conv_inputs, [1, -1, conv_inputs.shape[2], conv_inputs.shape[3]]) # Fused => reshape minibatch to convolution groups.
      # print("conv_inputs after reshaping", conv_inputs)

      # print('weights before reshaping: ', weights)
      weights = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])
      # print('weights after reshaping: ', weights)

      # Convolve
      padding = 0
      kernel = self.kernel_size[0]
      resample_kernel = [1,3,3,1]
      data_format = 'NHWC' #'NCHW'
      if self.sampling == 'up':
          # print('up')
          x = upsample_conv_2d_grouped(conv_inputs, weights, data_format=data_format, k=resample_kernel, padding=padding)
      else:
          padding_mode = {0: 'SAME', -(kernel // 2): 'VALID'}[padding]
          x = tf.nn.conv2d(conv_inputs, weights, data_format=data_format, strides=[1, 1, 1, 1], padding=padding_mode)
      out_shape = [-1,
                   inputs[0].shape[1] * 2 if self.sampling == 'up' else inputs[0].shape[1],
                   inputs[0].shape[2] * 2 if self.sampling == 'up' else inputs[0].shape[2],
                   style.shape[0],
                   self.filters,
                   ]
      # print(x)
      x = tf.reshape(x, out_shape)  # Fused => reshape convolution groups back to minibatch.
      # print(x)
      x = tf.transpose(x, [0, 3, 1, 2, 4])
      # x = tf.transpose(x, [0, 2, 3, 4, 1])
      # print(x)
      # print(x)
      # print(x)

      ##############################
      if self.noise:
          noise = tf.random.normal([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1, 1], dtype=x.dtype)
          x += noise*self.noise_strength/2
      # print(x)
      x = nn.bias_add(x, self.bias, data_format=data_format)
      # print(x)
      # 1 / 0
      if self.act == 'lrelu':
        x = tf.nn.leaky_relu(x, alpha=0.2)*tf.math.sqrt(2.0)
      elif self.act == 'linear' or self.act is None:
        pass
      else:
        raise ValueError('Activation is unsupported.')
      return x

  def compute_output_shape(self, input_shape):
    input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(ModConv2d_grouped, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding
