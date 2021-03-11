import pickle
import PIL
from PIL.PngImagePlugin import PngImageFile, PngInfo

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from layers.other import float_img_to_int8
from networks.load_network import load_network, get_const_layer

SNAPSHOT_MAIN_PATH = "networks/181220-0240-0338_panels.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_2019-04-30-stylegan-danbooru2018-portraits-02095-066083.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_HorseF-Emoji-0040-Pokemon-0040-Simpsons-0016-Pokemon-000032.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_pixelation-003-1080.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_TL-0-290-18fid.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_TL-0-290-18fid-TP-w.pkl"
# SNAPSHOT_MAIN_PATH = "networks/monsters-6-387-17fid.pkl"


# TODO: улучшить склеивание рандомных семян

if __name__=="__main__":
    rnd = np.random.RandomState(1)

    seed_main_styles = [483871]
    for seed_main_style in seed_main_styles:
        TRUNCATION_PSI = 0.7
        seeds_structure_style = np.array([
            [7, 6, 3, 7, 6]
        ])

        # seeds_structure_style = np.random.randint(np.ones((2, 3)) * (2 << 20)) # np.ones((y, x))

        seeds_shape = seeds_structure_style.shape

        flat_seeds_structure_style = [item for sublist in seeds_structure_style for item in sublist]
        flat_seeds_structure_style.append(seed_main_style)
        n_styles = len(flat_seeds_structure_style)
        styles = np.zeros([n_styles] + [512])


        for i, structure_style in enumerate(flat_seeds_structure_style):
            styles[i] = np.random.RandomState(structure_style).randn(512)

        main_const = get_const_layer(SNAPSHOT_MAIN_PATH)
        x_step = main_const.shape[2]
        y_step = main_const.shape[1]
        main_const = np.concatenate([main_const]*seeds_shape[1], axis=2)
        main_const = np.concatenate([main_const]*seeds_shape[0], axis=1)
        # main_const = np.concatenate([main_const]*1, axis=0)

        style_masks = []
        for style_map_strength in range(6):
            shape_modifier = 2 ** style_map_strength
            style_mask = np.zeros([1,
                                   n_styles,
                                   main_const.shape[1] * shape_modifier,
                                   main_const.shape[2] * shape_modifier])
            i_style = 0
            for y_crd in range(0, style_mask.shape[2], y_step * shape_modifier):
                for x_crd in range(0, style_mask.shape[3], x_step * shape_modifier):
                    style_mask[:, i_style,
                    y_crd: y_crd + y_step*shape_modifier,
                    x_crd: x_crd + x_step*shape_modifier] += 1.0
                    i_style += 1

            # # gaussian mask smoothing
            # for i in range(n_styles-1):
            #     style_mask[0, i] = cv2.blur(style_mask[0, i], (3, 3), 0)
            # # normalize probs
            # style_mask /= style_mask.sum(axis=(0, 1), keepdims=True)
            # assert np.abs(style_mask.sum(axis=(0, 1), keepdims=True).max() - 1) < 1e-6 and\
            #        np.abs(style_mask.sum(axis=(0, 1), keepdims=True).min() - 1) < 1e-6

            style_masks.append(style_mask)

        print('Loading network...')
        main_model = load_network(SNAPSHOT_MAIN_PATH, main_const.shape[1:3], middle_input_synth=None, trunc_psi=TRUNCATION_PSI, n_styles=n_styles)
        print('Predicting main...')
        main_images = main_model.predict([styles[np.newaxis], main_const] + style_masks)

        main_images = float_img_to_int8(main_images)
        main_images = main_images[-1][0]

        # im_min_px = cv2.resize(np.array(main_images), (int(np.array(main_images).shape[1] // 2),
        #                                                int(np.array(main_images).shape[0] // 2)),
        #                        interpolation=cv2.INTER_NEAREST)
        # im_4px = cv2.resize(np.array(im_min_px), (int(np.array(im_min_px).shape[1] * 4),
        #                                           int(np.array(im_min_px).shape[0] * 4)),
        #                        interpolation=cv2.INTER_NEAREST)
        # main_images = im_4px


        main_image_ = PIL.Image.fromarray(main_images, 'RGB')
        plt.imshow(main_image_)
        plt.show()
        if True:
            print(seed_main_style)
            print(seeds_structure_style)
            metadata = PngInfo()
            metadata.add_text(key='gen_info', value="seeds: " + str(seeds_structure_style) + ' psi: ' + str([TRUNCATION_PSI]))
            main_image_.save(
                f'imgs/seed_{seed_main_style}-shape{main_const.shape[1]}x{main_const.shape[2]}-{np.random.randint(1, 99999)}_sc{4}.png',
                pnginfo=metadata)

    print("Success!")


# from PIL import Image
# image_name = "seed_88744-shape8x12-70908_sc4"
# im = Image.open(f'imgs/{image_name}.png')  # Needed only for .png EXIF data (see citation above)
# im.load()  # Needed only for .png EXIF data (see citation above)
# print(im.info['gen_info'])
