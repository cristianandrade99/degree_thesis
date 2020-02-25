import tensorflow as tf
from tensorflow.keras import layers
import custom_layers as cl

# Dictionary keys
enc_fin_den_len_k = "f_den_len"
dec_den_info_k = "dec_den_info"

enc_dec_lys_info_k = "enc_dec_lys_info"
input_shape_k = "input_shape"

# Activations
r_act = "ReLU"
lr_act = "leakyReLU"
th_act = "tanh"
sgm_act = "sigmoid"

# It creates a encoder architecture in a keras model
def encoder_module(enc_info):
    model = tf.keras.Sequential()

    fst_layer_created = False
    input_shape = enc_info[input_shape_k]

    for depth, use_bn, act, f, s in enc_info[enc_dec_lys_info_k]:

        actual_layer = layers.Conv2D(depth, (f, f), strides=(s, s), padding='same', use_bias=not use_bn)

        if not fst_layer_created:
            actual_layer = layers.Conv2D(depth, (f, f), strides=(s, s), padding='same', use_bias=not use_bn, input_shape=input_shape)
            fst_layer_created = True

        model.add(actual_layer)

        if use_bn:
            model.add(layers.BatchNormalization())

        if act != None:
            model.add(keras_activation_layer(act))

    model.add(layers.Flatten())
    model.add(layers.Dense(enc_info[enc_fin_den_len_k]))

    return model

# It creates a decoder architecture in a keras model
def decoder_module(dec_info):
    model = tf.keras.Sequential()

    den_shs, use_bn, act, input_shape = dec_info[dec_den_info_k]

    model.add(layers.Dense(den_shs[0]*den_shs[1]*den_shs[2], use_bias= not use_bn, input_shape=input_shape) )

    if use_bn:
        model.add(layers.BatchNormalization())

    if act != None:
        model.add(keras_activation_layer(act))

    model.add(layers.Reshape((den_shs[0],den_shs[1],den_shs[2])))

    for depth, use_bn, act, f, s in dec_info[enc_dec_lys_info_k]:
        model.add( layers.Conv2DTranspose(depth, (f,f),strides=(s, s),padding='same', use_bias=not use_bn) )

        if use_bn:
            model.add(layers.BatchNormalization())

        if act != None:
            model.add(keras_activation_layer(act))

    return model

def keras_activation_layer(act):
    activ = None
    if act == r_act:
        activ = layers.ReLU()
    elif act == lr_act:
        activ = layers.LeakyReLU()
    elif act == th_act:
        activ = cl.Tanh()
    elif act == sgm_act:
        activ = cl.Sigmoid()
    return activ
