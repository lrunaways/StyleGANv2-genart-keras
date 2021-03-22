import pickle
import PIL

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from layers.other import Dense, normalize_2nd_moment, upsample, float_img_to_int8, Dlatent_avg
from layers.ModConv2d import ModConv2d, torgb
from layers.Block import Block
from StyleGAN import StyleGAN

def load_network(snapshot_path, const_layer_shape, n_styles, middle_input_synth=None, trunc_psi=0.7):
    with open(snapshot_path, 'rb') as f:
        weights = pickle.load(f)
    #TODO: auto resolution and n_dense infeence
    model = StyleGAN(const_shape=const_layer_shape, n_dense=2, resolution=256, middle_input_synth=middle_input_synth, n_styles=n_styles)
    layers_names = []
    for layer in model.layers:
        layers_names.append(layer.name)

    weights_to_idx = {'weight': 0, 'bias': 1, 'mod_weight': 2, 'mod_bias': 3, 'noise_strength': 4}
    for weight_name in weights.keys():
        weight_name_splitted = weight_name.split('/')
        if weight_name_splitted[0] == 'G_synthesis':
            try:
                if weight_name_splitted[1][:5] == 'noise':
                    continue
                if weight_name_splitted[2] == 'Const':
                    continue
                idx = weights_to_idx[weight_name_splitted[-1]]
                layer = model.get_layer(weight_name[len(weight_name_splitted[0]) + 1: -len(weight_name_splitted[-1]) - 1])
                layer.variables[idx].assign([weights[weight_name]] if idx == 4 else weights[weight_name])
                print('G_synthesis:', weight_name, ':', layer.variables[idx].name)
                # pass
            except Exception:
                print('G_synthesis: ', weight_name, ': BAD')
        elif weight_name_splitted[0] == 'G_mapping':
            try:
                idx = weights_to_idx[weight_name_splitted[-1]]
                layer = model.get_layer(weight_name[len(weight_name_splitted[0]) + 1: -len(weight_name_splitted[-1]) - 1])
                print('G_mapping: ', weight_name, ':', layer.variables[idx].name)
                layer.variables[idx].assign([weights[weight_name]] if idx == 4 else weights[weight_name])
            except Exception:
                print('G_mapping: ', weight_name, ': BAD')
        elif weight_name_splitted[0] == 'dlatent_avg':
            layer = model.get_layer(weight_name_splitted[0])
            print('dlatent_avg: ', weight_name, ':', layer.variables[0].name)
            layer.variables[0].assign([weights[weight_name]])
    return model

def get_const_layer(snapshot_path):
    with open(snapshot_path, 'rb') as f:
        weights = pickle.load(f)['G_synthesis/4x4/Const/const']
    const_layer = np.transpose(weights, (0, 2, 3, 1))  #(1, None, None, 512))
    return const_layer
