import pickle
import PIL

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from layers.other import float_img_to_int8
from networks.load_network import load_network, get_const_layer

SNAPSHOT_MAIN_PATH = "networks/181220-0240-0338_panels.pkl"
# SNAPSHOT_MAIN_PATH = "networks/maps.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_pixelation-002-000774.pkl"
# SNAPSHOT_SUB_PATH = "networks/monsters-6-387-17fid.pkl"


# TODO: групповые конволюции
# TODO: улучшить склеивание рандомных семян

if __name__=="__main__":
    # tf.keras.utils.plot_model(
    #     main_model, to_file='networks/main_model.png', show_shapes=False,
    #     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
    #
    # for layer in main_model.layers:
    #     print(layer.name)
    rnd = np.random.RandomState(1)
    for i_im in range(100):
        seed_main_style = rnd.randint(2 ** 30)
        seeds_structure_style = np.array([[1, 2, 3, 4],
                                          [2, 3, 4, 5]])
        seeds_shape = seeds_structure_style.shape

        main_input_z = np.random.RandomState(7).randn(1, 512)

        flat_seeds_structure_style = [item for sublist in seeds_structure_style for item in sublist]
        styles = np.zeros([len(flat_seeds_structure_style)] + [512])
        for i, structure_style in enumerate(flat_seeds_structure_style):
            styles[i] = np.random.RandomState(structure_style).randn(512)
        main_input_z = np.concatenate([main_input_z, np.random.RandomState(rnd.randint(2**30)).randn(2, 512)], axis=0)

        main_const = get_const_layer(SNAPSHOT_MAIN_PATH)

        # sub_const = get_const_layer(SNAPSHOT_SUB_PATH)
        # main_const = np.concatenate([main_const[:,:1,:,:], main_const], axis=1)
        main_const = np.concatenate([main_const]*4, axis=2)
        main_const = np.concatenate([main_const]*2, axis=1)
        main_const = np.concatenate([main_const]*1, axis=0)

        # main_const = tf.image.resize(
        #     main_const, (main_const.shape[1] * 3, main_const.shape[2] * 3), method=tf.image.ResizeMethod.GAUSSIAN,
        #     preserve_aspect_ratio=False, antialias=False, name=None)
        # main_const = =
        n_styles = len(styles)
        style_mask_4 = np.zeros([1, n_styles] + list(main_const.shape[1:-1]))
        style_mask_4[:, 0, :, :] += 1.0
        # style_mask_4[:, 0, 0:4, 0:4] += 1.0
        # style_mask_4[:, 1, 0:4, 4:8] += 1.0
        # style_mask_4[:, 2, 0:4, 8:12] += 1.0
        # style_mask_4[:, 3, 0:4, 12:16] += 1.0
        # style_mask_4[:, 4, 4:8, 0:4] += 1.0
        # style_mask_4[:, 5, 4:8, 4:8] += 1.0
        # style_mask_4[:, 6, 4:8, 8:12] += 1.0
        # style_mask_4[:, 7, 4:8, 12:16] += 1.0
        # style_mask_4

        style_mask_8 = np.zeros([1, n_styles] + [x*2 for x in main_const.shape[1:-1]])
        style_mask_8[:, 0, :, :] += 1.0

        main_model = load_network(SNAPSHOT_MAIN_PATH, main_const.shape[1:3], middle_input_synth=None, n_styles=n_styles)
        print('Predicting main...')
        main_images = main_model.predict([styles[np.newaxis], main_const, style_mask_4, style_mask_8])

        # if False:
        #     sub_model = load_network(SNAPSHOT_SUB_PATH, sub_const.shape[1:3],
        #                              middle_input_synth=[128, 128, 128,64])  # [in_shape_x, in_shape_y, in_shape_ch, res_level]
        #     for i, out in enumerate(main_images):
        #         print(i, out.shape)
        #     i = 8
        #     sub_images = sub_model.predict([sub_input_z, main_images[i], main_images[i+1]])
        #     sub_images = float_img_to_int8(sub_images)
        #     sub_images[-1][0].save(f'imgs/seed_{sub_seed}-mainseed_{main_seed}.png')


        main_images = float_img_to_int8(main_images)
        main_images = main_images[-1][0]

        im_min_px = cv2.resize(np.array(main_images), (int(np.array(main_images).shape[1] // 2),
                                                       int(np.array(main_images).shape[0] // 2)),
                               interpolation=cv2.INTER_NEAREST)
        im_4px = cv2.resize(np.array(im_min_px), (int(np.array(im_min_px).shape[1] * 4),
                                                  int(np.array(im_min_px).shape[0] * 4)),
                               interpolation=cv2.INTER_NEAREST)
        main_images = im_4px

        main_image_ = PIL.Image.fromarray(main_images, 'RGB')
        # plt.imshow(main_image_)
        # plt.show()
        if True:
            main_image_.save(f'imgs/seed_{seed_main_style}-shape{main_const.shape[1]}x{main_const.shape[2]}.png')

    print("Success!")


# import pickle
# import PIL
#
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# from layers.other import float_img_to_int8
# from networks.load_network import load_network, get_const_layer
#
# # SNAPSHOT_MAIN_PATH = "networks/181220-0240-0338_panels.pkl"
# # SNAPSHOT_MAIN_PATH = "networks/maps.pkl"
# SNAPSHOT_MAIN_PATH = "networks/dict_pixelation-002-000774.pkl"
# # SNAPSHOT_SUB_PATH = "networks/monsters-6-387-17fid.pkl"
#
#
# # TODO: групповые конволюции
# # TODO: улучшить склеивание рандомных семян
#
# if __name__=="__main__":
#     # tf.keras.utils.plot_model(
#     #     main_model, to_file='networks/main_model.png', show_shapes=False,
#     #     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
#     #
#     # for layer in main_model.layers:
#     #     print(layer.name)
#     rnd = np.random.RandomState(1)
#     for i_im in range(100):
#         seed_main_style = rnd.randint(2 ** 30)
#         seeds_structure_style = np.array([[1, 2, 3, 4],
#                                           [2, 3, 4, 5]])
#         seeds_shape = seeds_structure_style.shape
#
#         main_input_z = np.random.RandomState(7).randn(1, 512)
#
#         flat_seeds_structure_style = [item for sublist in seeds_structure_style for item in sublist]
#         styles = np.zeros([len(flat_seeds_structure_style)] + [512])
#         for i, structure_style in enumerate(flat_seeds_structure_style):
#             styles[i] = np.random.RandomState(structure_style).randn(512)
#         main_input_z = np.concatenate([main_input_z, np.random.RandomState(rnd.randint(2**30)).randn(2, 512)], axis=0)
#
#         main_const = get_const_layer(SNAPSHOT_MAIN_PATH)
#
#         # sub_const = get_const_layer(SNAPSHOT_SUB_PATH)
#         # main_const = np.concatenate([main_const[:,:1,:,:], main_const], axis=1)
#         main_const = np.concatenate([main_const, main_const], axis=2)
#         main_const = np.concatenate([main_const]*3, axis=0)
#
#         # main_const = tf.image.resize(
#         #     main_const, (main_const.shape[1] * 3, main_const.shape[2] * 3), method=tf.image.ResizeMethod.GAUSSIAN,
#         #     preserve_aspect_ratio=False, antialias=False, name=None)
#         # main_const = =
#
#         main_model = load_network(SNAPSHOT_MAIN_PATH, (4, 8), middle_input_synth=None)
#         print('Predicting main...')
#         main_images = main_model.predict([main_input_z, main_const])
#
#         # if False:
#         #     sub_model = load_network(SNAPSHOT_SUB_PATH, sub_const.shape[1:3],
#         #                              middle_input_synth=[128, 128, 128,64])  # [in_shape_x, in_shape_y, in_shape_ch, res_level]
#         #     for i, out in enumerate(main_images):
#         #         print(i, out.shape)
#         #     i = 8
#         #     sub_images = sub_model.predict([sub_input_z, main_images[i], main_images[i+1]])
#         #     sub_images = float_img_to_int8(sub_images)
#         #     sub_images[-1][0].save(f'imgs/seed_{sub_seed}-mainseed_{main_seed}.png')
#
#
#         main_images = float_img_to_int8(main_images)
#         main_images = main_images[-1][0]
#
#         im_min_px = cv2.resize(np.array(main_images), (int(np.array(main_images).shape[1] // 2),
#                                                        int(np.array(main_images).shape[0] // 2)),
#                                interpolation=cv2.INTER_NEAREST)
#         im_4px = cv2.resize(np.array(im_min_px), (int(np.array(im_min_px).shape[1] * 4),
#                                                   int(np.array(im_min_px).shape[0] * 4)),
#                                interpolation=cv2.INTER_NEAREST)
#         main_images = im_4px
#
#         main_image_ = PIL.Image.fromarray(main_images, 'RGB')
#         # plt.imshow(main_image_)
#         # plt.show()
#         if True:
#             main_image_.save(f'imgs/seed_{seed_main_style}-shape{main_const.shape[1]}x{main_const.shape[2]}.png')
#
#     print("Success!")

