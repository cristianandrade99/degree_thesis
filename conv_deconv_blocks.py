import tensorflow as tf
import custom_layers as cl

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
