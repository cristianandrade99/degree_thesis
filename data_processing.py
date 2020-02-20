import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

N_H = 64
N_W = 64
N_C = 1

def load_process_fp_dataset(data_dir_patt,input_shape_k,batch_size):
    global N_H,N_W,N_C

    ds_data_dirs = tf.data.Dataset.list_files("./"+data_dir_patt[0]+"/*"+data_dir_patt[1],shuffle=False)

    data_list = list(ds_data_dirs.as_numpy_iterator())
    print("Fingerprints Loaded:",len(data_list),"from:",data_dir_patt[0])

    N_H = input_shape_k[0]
    N_W = input_shape_k[1]
    N_C = input_shape_k[2]

    ds_data_dirs = ds_data_dirs.map(load_process_images, num_parallel_calls=AUTOTUNE)

    ds_data_dirs = ds_data_dirs.shuffle(buffer_size=128)
    ds_data_dirs = ds_data_dirs.batch(batch_size)
    ds_data_dirs = ds_data_dirs.prefetch(buffer_size=AUTOTUNE)

    data_list_batched = list(ds_data_dirs.as_numpy_iterator())
    print("Batches Created:",len(data_list_batched))
    print("shape of batch:",data_list_batched[0].shape)

    return ds_data_dirs

def load_process_images(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img)
    if N_C == 1:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [N_H, N_W],preserve_aspect_ratio=False)
    img = tf.cast(img,tf.float32)
    img = img*(2.0/255.0)-1.0

    return img

def load_verification_images(n_h,n_w,n_c):
    global N_H,N_W,N_C

    N_H = n_h
    N_W = n_w
    N_C = n_c

    validation_images_source = "./Validation_images/*.png"
    ds_data_dirs = tf.data.Dataset.list_files(validation_images_source,shuffle=False)
    ds_data_dirs = ds_data_dirs.map(load_process_images, num_parallel_calls=AUTOTUNE)
    ds_data_dirs = ds_data_dirs.batch(10)
    ds = None
    for v in ds_data_dirs:
        ds = v

    return ds
