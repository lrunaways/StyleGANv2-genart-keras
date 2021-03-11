import pickle
import PIL

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from layers.other import *
from layers.ModConv2d import ModConv2d, ModConv2d_grouped, torgb
from layers.Block import Block

def StyleGAN(const_shape, resolution, n_dense=2, n_styles=5, middle_input_synth=None, TRUNCATION_PSI=0.7):
    """

    :param const_shape:
    :param middle_input_synth: None - full network; 8, 16, ... - input resolution
    :return:
    """
    def G_mapping(input_, mapping_layers, dlatent_broadcast, dlatens_size):
        styles_shape = input_.shape[1]
        dlatent_avg = Dlatent_avg(512)(input_)
        x = normalize_2nd_moment(input_)
        x = tf.reshape(x, (-1, 512))
        for layer_idx in range(mapping_layers):
            x = Dense(dlatens_size, lrmul=0.01, name=f'Dense{layer_idx}')(x)
        x = tf.reshape(x, (-1, styles_shape, 512))
        x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1, 1])
        x = lerp(x, dlatent_avg, TRUNCATION_PSI)
        return x

    def G_synthesis(W, const_layer, style_strength_maps):
        data_format = 'NHWC'
        def nf(stage):
            f = [512, 512, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            return f[stage-1]
        # x = ModConv2d(rank=2, filters=512, kernel_size=3, sampling='down', padding='same')([base_res, W])

        resolution_log2 = int(np.log2(resolution))

        # num_layers = resolution_log2 * 2 - 2

        # if middle_input_synth is None:
        # Layers for 4x4 resolution.
        middle_input = None

        y = None
        x = ModConv2d_grouped(rank=2, name="4x4/Conv", kernel_size=3, sampling=None, filters=nf(1), padding='same')([const_layer, W[:, 0]])
        x = tf.reduce_sum(x*style_strength_maps[0], axis=1)
        y = torgb(x, y, W[:, 0], res_name='4x4', is_grouped=True, style_strength_map=style_strength_maps[0])
        outs = []

        # Layers for >=8x8 resolutions.
        for res in range(3, resolution_log2 + 1):
            print(f'{2 ** res}x{2 ** res}')
            if res <= 3:
                is_grouped = True
                style_strength_map = style_strength_maps[res - 3 + 1]
            else:
                is_grouped = False
                style_strength_map = None
            x = Block(x, latents=W[:, 0], fmaps=nf(res-2), res_name=f'{2 ** res}x{2 ** res}', is_grouped=is_grouped, style_strength_map=style_strength_map)
            y = upsample(y, [1, 3, 3, 1], data_format, res_name=f'{2 ** res}x{2 ** res}')
            y = torgb(x, y, W[:, 0], res_name=f'{2 ** res}x{2 ** res}', is_grouped=is_grouped, style_strength_map=style_strength_map)
            outs.append(x)
            outs.append(y)
        return outs, middle_input
    #
    input_z = tf.keras.layers.Input(shape=[n_styles, 512], name='input_z')
    input_const = tf.keras.layers.Input(shape=[const_shape[0], const_shape[1], 512], name='input_const')

    n_style_strength_maps = 7  # int(np.log2(resolution))
    style_strength_maps = []
    style_strength_maps_inputs = []
    for style_map_strength in range(n_style_strength_maps - 1):
        shape_modifier = 2**style_map_strength
        input_style_map = tf.keras.layers.Input(shape=[n_styles,
                                                       const_shape[0] * shape_modifier,
                                                       const_shape[1] * shape_modifier],
                                  name='style_map_strength_' + str(4*shape_modifier))
        input_style_newaxis = input_style_map[..., np.newaxis]
        style_strength_maps_inputs.append(input_style_map)
        style_strength_maps.append(input_style_newaxis)
    W = G_mapping(input_z, mapping_layers=n_dense, dlatent_broadcast=14, dlatens_size=512)
    synth_out, middle_input = G_synthesis(W, input_const, style_strength_maps)

    if middle_input is None:
        input_ = [input_z, input_const] + style_strength_maps_inputs
    else:
        input_ = [input_z] + middle_input
    return tf.keras.Model(inputs=input_, outputs=synth_out)