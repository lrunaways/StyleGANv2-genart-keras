# G                             Params    OutputShape         WeightShape
# ---                           ---       ---                 ---
# latents_in                    -         (?, 512)            -
# labels_in                     -         (?, 0)              -
# G_mapping/Normalize           -         (?, 512)            -
# G_mapping/Dense0              262656    (?, 512)            (512, 512)
# G_mapping/Dense1              262656    (?, 512)            (512, 512)
# G_mapping/Broadcast           -         (?, 14, 512)        -
# dlatent_avg                   -         (512,)              -
# Truncation/Lerp               -         (?, 14, 512)        -
# G_synthesis/4x4/Const         8192      (?, 512, 4, 4)      (1, 512, 4, 4)
# G_synthesis/4x4/Conv          2622465   (?, 512, 4, 4)      (3, 3, 512, 512)
# G_synthesis/4x4/ToRGB         264195    (?, 3, 4, 4)        (1, 1, 512, 3)
# G_synthesis/8x8/Conv0_up      2622465   (?, 512, 8, 8)      (3, 3, 512, 512)
# G_synthesis/8x8/Conv1         2622465   (?, 512, 8, 8)      (3, 3, 512, 512)
# G_synthesis/8x8/Upsample      -         (?, 3, 8, 8)        -
# G_synthesis/8x8/ToRGB         264195    (?, 3, 8, 8)        (1, 1, 512, 3)
# G_synthesis/16x16/Conv0_up    2622465   (?, 512, 16, 16)    (3, 3, 512, 512)
# G_synthesis/16x16/Conv1       2622465   (?, 512, 16, 16)    (3, 3, 512, 512)
# G_synthesis/16x16/Upsample    -         (?, 3, 16, 16)      -
# G_synthesis/16x16/ToRGB       264195    (?, 3, 16, 16)      (1, 1, 512, 3)
# G_synthesis/32x32/Conv0_up    2622465   (?, 512, 32, 32)    (3, 3, 512, 512)
# G_synthesis/32x32/Conv1       2622465   (?, 512, 32, 32)    (3, 3, 512, 512)
# G_synthesis/32x32/Upsample    -         (?, 3, 32, 32)      -
# G_synthesis/32x32/ToRGB       264195    (?, 3, 32, 32)      (1, 1, 512, 3)
# G_synthesis/64x64/Conv0_up    1442561   (?, 256, 64, 64)    (3, 3, 512, 256)
# G_synthesis/64x64/Conv1       721409    (?, 256, 64, 64)    (3, 3, 256, 256)
# G_synthesis/64x64/Upsample    -         (?, 3, 64, 64)      -
# G_synthesis/64x64/ToRGB       132099    (?, 3, 64, 64)      (1, 1, 256, 3)
# G_synthesis/128x128/Conv0_up  426369    (?, 128, 128, 128)  (3, 3, 256, 128)
# G_synthesis/128x128/Conv1     213249    (?, 128, 128, 128)  (3, 3, 128, 128)
# G_synthesis/128x128/Upsample  -         (?, 3, 128, 128)    -
# G_synthesis/128x128/ToRGB     66051     (?, 3, 128, 128)    (1, 1, 128, 3)
# G_synthesis/256x256/Conv0_up  139457    (?, 64, 256, 256)   (3, 3, 128, 64)
# G_synthesis/256x256/Conv1     69761     (?, 64, 256, 256)   (3, 3, 64, 64)
# G_synthesis/256x256/Upsample  -         (?, 3, 256, 256)    -
# G_synthesis/256x256/ToRGB     33027     (?, 3, 256, 256)    (1, 1, 64, 3)
# ---                           ---       ---                 ---
# Total                         23191522


def layer(x, layer_idx, fmaps, kernel, up=False):
    x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up,
                               resample_kernel=resample_kernel, fused_modconv=fused_modconv)
    if use_noise:
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
    return apply_bias_act(x, act=act, clamp=conv_clamp)


def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, lrmul=1, fused_modconv=False, trainable=True, use_spectral_norm=False):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    wshape = [kernel, kernel, x.shape[1].value, fmaps]
    w = get_weight(wshape, lrmul=lrmul, trainable=trainable, use_spectral_norm=use_spectral_norm)
    if x.dtype.name == 'float16' and not fused_modconv and demodulate:
        w *= np.sqrt(1 / np.prod(wshape[:-1])) / tf.reduce_max(tf.abs(w), axis=[0,1,2]) # Pre-normalize to avoid float16 overflow.
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var='mod_weight', trainable=trainable, use_spectral_norm=use_spectral_norm) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var='mod_bias', trainable=trainable) + 1 # [BI] Add bias (initially 1).
    if x.dtype.name == 'float16' and not fused_modconv and demodulate:
        s *= 1 / tf.reduce_max(tf.abs(s)) # Pre-normalize to avoid float16 overflow.
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # 2D convolution.
    x = conv2d(x, tf.cast(w, x.dtype), up=up, down=down, resample_kernel=resample_kernel)

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x