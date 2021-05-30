import pickle
import PIL

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from layers.other import Dense, normalize_2nd_moment, upsample, float_img_to_int8, Dlatent_avg
from layers.ModConv2d import ModConv2d, torgb
from layers.Block import Block
from StyleGAN import StyleGAN

def load_network(snapshot_path, const_layer_shape, n_styles, middle_input_synth=None, trunc_psi=0.7, microstyle_layer=6):
    with open(snapshot_path, 'rb') as f:
        weights = pickle.load(f)
    # inference resolution and number of dense layers
    all_resolutions = [x.split('/')[1] for x in list(weights.keys()) if x.find('RGB/bias') != -1]
    max_resolution = max([int(x.split('x')[0]) for x in all_resolutions])
    all_dense = [x.split('/')[1] for x in list(weights.keys()) if x.find('Dense') != -1]
    max_dense = max([int(x.split('Dense')[1]) for x in all_dense]) + 1

    model = StyleGAN(const_shape=const_layer_shape, n_dense=max_dense, resolution=max_resolution, middle_input_synth=middle_input_synth,
                     n_styles=n_styles, microstyle_layer=microstyle_layer)
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
                input_weights = [weights[weight_name]] if idx == 4 else weights[weight_name]
                layer.variables[idx].assign(input_weights)
                print('G_synthesis:', weight_name, ':', layer.variables[idx].name)
                # pass
            except ValueError as e:
                print('G_synthesis: ', weight_name, ': BAD')
                print(f'\tnetwork: {layer.variables[idx].shape}, input: {input_weights.shape}')
                pass
        elif weight_name_splitted[0] == 'G_mapping':
            try:
                idx = weights_to_idx[weight_name_splitted[-1]]
                layer = model.get_layer(weight_name[len(weight_name_splitted[0]) + 1: -len(weight_name_splitted[-1]) - 1])
                input_weights = [weights[weight_name]] if idx == 4 else weights[weight_name]
                layer.variables[idx].assign([weights[weight_name]] if idx == 4 else weights[weight_name])
                print('G_mapping: ', weight_name, ':', layer.variables[idx].name)
            except Exception:
                print('G_mapping: ', weight_name, ': BAD')
                print(f'\tnetwork: {layer.variables[idx].shape}, input: {input_weights.shape}')
                pass
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
