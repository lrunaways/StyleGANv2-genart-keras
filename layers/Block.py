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
from layers.ModConv2d import ModConv2d, ModConv2d_grouped

# Main block for one resolution.
def Block(x, latents, fmaps, res_name, is_grouped, style_strength_map=None):  # res = 3..resolution_log2
    if not is_grouped:
        x = ModConv2d(rank=2, filters=fmaps, sampling='up', kernel_size=3, noise=True, demodulate=True,
                        name=str(res_name) + '/Conv0_up')([x, latents[0:1, -1]])
        x = ModConv2d(rank=2, filters=fmaps, sampling=None, kernel_size=3, noise=True, demodulate=True,
                        name=str(res_name) + '/Conv1')([x, latents[0:1, -1]])
    else:
        x = ModConv2d_grouped(rank=2, filters=fmaps, sampling='up', kernel_size=3, noise=True, demodulate=True,
                        name=str(res_name) + '/Conv0_up')([x, latents])
        x = tf.reduce_sum(x * style_strength_map, axis=1)
        x = ModConv2d_grouped(rank=2, filters=fmaps, sampling=None, kernel_size=3, noise=True, demodulate=True,
                        name=str(res_name) + '/Conv1')([x, latents])
        x = tf.reduce_sum(x * style_strength_map, axis=1)
    return x


# # Main block for one resolution.
# def Block(x, latents, fmaps, res_name):  # res = 3..resolution_log2
#     t = x
#     x = ModConv2d(rank=2, filters=fmaps, sampling='up', kernel_size=3, noise=True, demodulate=True, name=str(res_name)+'/Conv0_up')([x, latents])
#     # with tf.variable_scope('Conv1'):True
#     x = ModConv2d(rank=2, filters=fmaps, sampling=None, kernel_size=3, noise=True, demodulate=True, name=str(res_name)+'/Conv1')([x, latents])
#     # if architecture == 'resnet':
#     #     with tf.variable_scope('Skip'):
#     #         t = conv2d_layer(t, fmaps=nf(res - 1), kernel=1, up=True, resample_kernel=resample_kernel)
#     #         x = (x + t) * (1 / np.sqrt(2))
#     return x
# import numpy as np
# import tensorflow as tf
#
# from tensorflow.python.eager import context
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.keras import activations
# from tensorflow.python.keras import backend
# from tensorflow.python.keras import constraints
# from tensorflow.python.keras import initializers
# from tensorflow.python.keras import regularizers
# from tensorflow.python.keras.engine.base_layer import Layer
# from tensorflow.python.keras.engine.input_spec import InputSpec
# # imports for backwards namespace compatibility
# # pylint: disable=unused-import
# from tensorflow.python.keras.layers.pooling import AveragePooling1D
# from tensorflow.python.keras.layers.pooling import AveragePooling2D
# from tensorflow.python.keras.layers.pooling import AveragePooling3D
# from tensorflow.python.keras.layers.pooling import MaxPooling1D
# from tensorflow.python.keras.layers.pooling import MaxPooling2D
# from tensorflow.python.keras.layers.pooling import MaxPooling3D
# # pylint: enable=unused-import
# from tensorflow.python.keras.utils import conv_utils
# from tensorflow.python.keras.utils import tf_utils
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import nn
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.util.tf_export import keras_export
#
# from upfirdn_2d import *
# from layers.other import Dense, normalize_2nd_moment
# from layers.ModConv2d import ModConv2d
#
# # Main block for one resolution.
# def Block(x, latents, fmaps, res_name):  # res = 3..resolution_log2
#     # t = x
#     first_mod_conv = ModConv2d(rank=2, filters=fmaps, sampling='up', kernel_size=3, noise=True, demodulate=True, name=str(res_name) + '/Conv0_up')
#     second_mod_conv = ModConv2d(rank=2, filters=fmaps, sampling=None, kernel_size=3, noise=True, demodulate=True, name=str(res_name) + '/Conv1')
#
#     # if res_name[0] == '8':
#     if res_name[0] in ['4', '8']:
#         x1 = first_mod_conv([x[:, :, :x.shape[2] // 2, :], latents[1:2]])
#         x1 = second_mod_conv([x1, latents[1:2]])
#         x2 = first_mod_conv([x[:, :, x.shape[2] // 2:, :], latents[2:3]])
#         x2 = second_mod_conv([x2, latents[2:3]])
#         x = tf.concat([x1, x2], axis=2)
#     else:
#         x = first_mod_conv([x, latents[0:1]])
#         x = second_mod_conv([x, latents[0:1]])
#
#     # x = second_mod_conv([x, latents])
#     return x
#
#
# # # Main block for one resolution.
# # def Block(x, latents, fmaps, res_name):  # res = 3..resolution_log2
# #     t = x
# #     x = ModConv2d(rank=2, filters=fmaps, sampling='up', kernel_size=3, noise=True, demodulate=True, name=str(res_name)+'/Conv0_up')([x, latents])
# #     # with tf.variable_scope('Conv1'):True
# #     x = ModConv2d(rank=2, filters=fmaps, sampling=None, kernel_size=3, noise=True, demodulate=True, name=str(res_name)+'/Conv1')([x, latents])
# #     # if architecture == 'resnet':
# #     #     with tf.variable_scope('Skip'):
# #     #         t = conv2d_layer(t, fmaps=nf(res - 1), kernel=1, up=True, resample_kernel=resample_kernel)
# #     #         x = (x + t) * (1 / np.sqrt(2))
# #     return x
