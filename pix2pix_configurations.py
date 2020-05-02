import tensorflow as tf
import keys as km

def paper_gen_disc_configuration():
    # [depth,n_f,apply_dropout on decoder]
    # n_f last decoder layer
    generator_config = [ [64,4,False],
                         [128,4,False],
                         [256,4,False],
                         [512,4,False],
                         [512,4,True],
                         [512,4,True],
                         [512,4,True],
                         [512,4,None],
                         4 ]

    # [depth, n_f, apply_batchnorm]
    discriminator_config_1 = [ [64,4,False],
                               [128,4,True],
                               [256,4,True] ]

    # stride equal to 1
    discriminator_config_2 = [ [512,4],
                                4 ]

    return create_dicc(generator_config,discriminator_config_1,discriminator_config_2)

def lenovo_gen_disc_configuration():
    # [depth,n_f,apply_dropout on decoder]
    # n_f last decoder layer
    generator_config = [ [1,4,False],
                         [1,4,False],
                         [1,4,False],
                         [1,4,False],
                         [1,4,True],
                         [1,4,True],
                         [1,4,True],
                         [1,4,None],
                         4 ]

    # [depth, n_f, apply_batchnorm]
    discriminator_config_1 = [ [1,4,False],
                               [1,4,True],
                               [1,4,True] ]

    # stride equal to 1
    discriminator_config_2 = [ [1,4],
                                4 ]

    return create_dicc(generator_config,discriminator_config_1,discriminator_config_2)

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

def create_dicc(generator_config,discriminator_config_1,discriminator_config_2):
    return {
        km.generator_config_k: generator_config,
        km.discriminator_config_1_k: discriminator_config_1,
        km.discriminator_config_2_k: discriminator_config_2
    }

def create_paper_gen_disc(gen_disc_config,fps_shape):

    # CONFIGURATION
    generator_config = gen_disc_config[km.generator_config_k]
    len_generator_config = len(generator_config)

    discriminator_config_1 = gen_disc_config[km.discriminator_config_1_k]
    discriminator_config_2 = gen_disc_config[km.discriminator_config_2_k]

    # GENERATOR
    inputs = tf.keras.layers.Input(fps_shape)
    x = inputs
    initializer = tf.random_normal_initializer(0.0,0.02)
    skips = []

    down_stack = []
    up_stack = []

    for i in range(len_generator_config-1):
        depth,n_f,apply_drop = generator_config[i]

        down_stack.append(downsample(depth,n_f,apply_batchnorm=False if i==0 else True))
        if (i<len_generator_config-2): up_stack.append(upsample(depth,n_f,apply_dropout=apply_drop))

    up_stack = reversed(up_stack)

    last = tf.keras.layers.Conv2DTranspose(fps_shape[2], generator_config[len_generator_config-1],strides=2,padding='same',kernel_initializer=initializer,activation='tanh')

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

    inp = tf.keras.layers.Input(shape=fps_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=fps_shape, name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])

    for depth,n_f,apply_bn in discriminator_config_1:
        x = downsample(depth, n_f,apply_bn)(x)

    x = tf.keras.layers.Conv2D(discriminator_config_2[0][0], discriminator_config_2[0][1], padding='same', strides=1,kernel_initializer=initializer,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1, discriminator_config_2[1], strides=1,kernel_initializer=initializer)(x)

    discriminator = tf.keras.Model(inputs=[inp, tar], outputs=x)

    return generator, discriminator
