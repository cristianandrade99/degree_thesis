import tensorflow as tf
import custom_layers as cl
import conv_deconv_models as md

# Dictionary keys
enc_fin_den_len_k = "f_den_len"
dec_den_info_k = "dec_den_info"

enc_lys_info_k = "enc_lys_info"
dec_lys_info_k = "dec_lys_info_k"
fps_shape_k = "fps_shape"

# Activations
r_act = "ReLU"
lr_act = "leakyReLU"
th_act = "tanh"
sgm_act = "sigmoid"

# It creates a encoder architecture in a keras model
def encoder_module(enc_info):

    inputs = tf.keras.layers.Input(enc_info[fps_shape_k])

    x = inputs
    for depth, use_bn, act, f, s in enc_info[enc_lys_info_k]:
        x = down_sample(depth,f,s,act,use_bn)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(enc_info[enc_fin_den_len_k])(x)

    return tf.keras.Model(inputs=inputs,outputs=x)

# It creates a decoder architecture in a keras model
def decoder_module(dec_info):

    den_shs, use_bn, act, input_shape = dec_info[dec_den_info_k]
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs

    x = tf.keras.layers.Dense(den_shs[0]*den_shs[1]*den_shs[2], use_bias= not use_bn, input_shape=input_shape)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if act != None:
        x = keras_activation_layer(act)(x)
    x = tf.keras.layers.Reshape((den_shs[0],den_shs[1],den_shs[2]))(x)

    for depth, use_bn, act, f, s in dec_info[dec_lys_info_k]:
        x = up_sample(depth,f,s,act,use_bn)(x)

    return tf.keras.Model(inputs=inputs,outputs=x)

# It creates a cvae for pix2pix model
def p2p_cvae_module(cvae_info):

    inputs = tf.keras.layers.input(cvae_info[fps_shape_k])
    x = inputs

    skips = []
    for depth, use_bn, act, f, s, use_drop, init in cvae_info[enc_lys_info_k]:
        x = down_sample(depth,use_bn,act,f,s,use_drop,init)(x)
        skips.append(x)

    dec_lys = []
    for depth, use_bn, act, f, s, use_drop, init in cvae_info[dec_lys_info_k]:
        dec_lys.append( up_sample(depth,use_bn,act,f,s,use_drop,init) )

    skips = reversed(skips[:-1])
    for lyr,skip in zip(dec_lys,skips):
        x = lyr(x)
        x = tf.keras.layers.Concatenate()([x,skip])

init_glorot = "glorot_uniform"
init_normal = "RandomNormal"

def down_sample(depth,f,s,act,use_bn=True,use_drop=[False,0.],init=[init_glorot,0.,0.]):
    block = tf.keras.Sequential()
    initializer = init_glorot if init[0]==init_glorot else tf.random_normal_initializer(init[1],init[2])
    block.add(tf.keras.layers.Conv2D(depth, (f,f), strides=(s,s), padding='same', use_bias=not use_bn, kernel_initializer=initializer))
    block = use_bn_drop_act(block,act,use_bn,use_drop)
    return block

def up_sample(depth,f,s,act,use_bn=True,use_drop=[False,0.],init=[init_glorot,0.,0.]):
    block = tf.keras.Sequential()
    initializer = init_glorot if init[0]==init_glorot else tf.random_normal_initializer(init[1],init[2])
    block.add(tf.keras.layers.Conv2DTranspose(depth, (f,f), strides=(s, s), padding='same', use_bias=not use_bn, kernel_initializer=initializer))
    block = use_bn_drop_act(block,act,use_bn,use_drop)
    return block

def use_bn_drop_act(block,act,use_bn=True,use_drop=[False,0.]):
    if use_bn:
        block.add(tf.keras.layers.BatchNormalization())
    if use_drop[0]:
        block.add(layers.Dropout(use_drop[1]))
    if act != None:
        block.add(keras_activation_layer(act))
    return block

def keras_activation_layer(act):
    activ = None
    if act == r_act:
        activ = tf.keras.layers.ReLU()
    elif act == lr_act:
        activ = tf.keras.layers.LeakyReLU(0.2)
    elif act == th_act:
        activ = cl.Tanh()
    elif act == sgm_act:
        activ = cl.Sigmoid()
    return activ

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0,0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0,0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def create_gen_disc(config):
    # GENERATOR
    inputs = tf.keras.layers.Input(config[md.fps_shape_k])
    x = inputs
    initializer = tf.random_normal_initializer(0.0,0.02)
    skips = []

    down_stack = [downsample(64, 4, apply_batchnorm=False),
                downsample(128, 4),
                downsample(256, 4),
                downsample(512, 4),
                downsample(512, 4),
                downsample(512, 4),
                downsample(512, 4),
                downsample(512, 4)]

    up_stack = [upsample(512, 4, apply_dropout=True),
              upsample(512, 4, apply_dropout=True),
              upsample(512, 4, apply_dropout=True),
              upsample(512, 4),
              upsample(256, 4),
              upsample(128, 4),
              upsample(64, 4)]

    last = tf.keras.layers.Conv2DTranspose(config[md.fps_shape_k][2], 4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    generator = tf.keras.Model(inputs=inputs, outputs=x)

    # DISCRIMINATOR
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=config[md.fps_shape_k], name='input_image')
    tar = tf.keras.layers.Input(shape=config[md.fps_shape_k], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)

    discriminator = tf.keras.Model(inputs=[inp, tar], outputs=last)

    return generator,discriminator

def create_gen_disc_lenovo(config):
    # GENERATOR
    inputs = tf.keras.layers.Input(config[md.fps_shape_k])
    x = inputs
    initializer = tf.random_normal_initializer(0.0,0.02)
    skips = []

    down_stack = [downsample(2, 4, apply_batchnorm=False),
                downsample(2, 4),
                downsample(2, 4),
                downsample(2, 4),
                downsample(2, 4),
                downsample(2, 4),
                downsample(2, 4),
                downsample(2, 4)]

    up_stack = [upsample(2, 4, apply_dropout=True),
              upsample(2, 4, apply_dropout=True),
              upsample(2, 4, apply_dropout=True),
              upsample(2, 4),
              upsample(2, 4),
              upsample(2, 4),
              upsample(2, 4)]

    last = tf.keras.layers.Conv2DTranspose(config[md.fps_shape_k][2], 4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    generator = tf.keras.Model(inputs=inputs, outputs=x)

    # DISCRIMINATOR
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=config[md.fps_shape_k], name='input_image')
    tar = tf.keras.layers.Input(shape=config[md.fps_shape_k], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(2, 4, False)(x)
    down2 = downsample(2, 4)(down1)
    down3 = downsample(2, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(2, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)

    discriminator = tf.keras.Model(inputs=[inp, tar], outputs=last)

    return generator,discriminator
