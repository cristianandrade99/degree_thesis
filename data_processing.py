import tensorflow as tf
import pathlib as pl
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

N_H = 28
N_W = 28
N_C = 1
N_C_loaded = 1

def load_process_fp_dataset(data_dir_patt,input_shape,batch_size):
    global N_H,N_W,N_C

    paterns = list_folder_patterns(data_dir_patt[0],data_dir_patt[1])
    ds_data_dirs_orig = tf.data.Dataset.list_files(paterns,shuffle=False)

    data_list = list(ds_data_dirs_orig.as_numpy_iterator())
    print("")
    print("Fingerprints Loaded:",len(data_list),"from:",data_dir_patt[0],"\n")

    N_H = input_shape[0]
    N_W = input_shape[1]
    N_C = input_shape[2]

    ds_data_dirs_orig = ds_data_dirs_orig.map(read_orig_images, num_parallel_calls=AUTOTUNE)
    ds_data_dirs_inv = ds_data_dirs_orig.map(convert_inv_images, num_parallel_calls=AUTOTUNE)

    ds_data_dirs = ds_data_dirs_orig.concatenate(ds_data_dirs_inv)

    #ds_data_dirs = ds_data_dirs.shuffle(buffer_size=128)
    ds_data_dirs = ds_data_dirs.batch(batch_size,True)
    ds_data_dirs = ds_data_dirs.prefetch(buffer_size=AUTOTUNE)

    data_list_batched = list(ds_data_dirs.as_numpy_iterator())
    print("Batches Created:",len(data_list_batched))
    print("shape of batch:",data_list_batched[0].shape,"\n")

    return ds_data_dirs

def load_verification_images(fps_shape,num_fps):
    global N_H,N_W,N_C

    N_H = fps_shape[0]
    N_W = fps_shape[1]
    N_C = fps_shape[2]

    validation_images_source = "./Img_Validation_images/*.png"
    ds_data_dirs = tf.data.Dataset.list_files(validation_images_source,shuffle=False)
    ds_data_dirs = ds_data_dirs.map(load_process_images, num_parallel_calls=AUTOTUNE)
    ds_data_dirs = ds_data_dirs.batch(10)
    ds = None
    for v in ds_data_dirs:
        ds = v

    return ds[0:num_fps,:]

def list_folder_patterns(root_dir,patern):
    patterns = []
    for root,folders,files in os.walk(root_dir):
        for file in files:
            file_dir = str(os.path.join(root,file))
            parent = str(pl.Path(file_dir).parent)
            act_patern = "./"+(str(parent)+"/*"+patern)
            if act_patern not in patterns:
                patterns.append(act_patern)
                continue
    return patterns

def read_orig_images(file_path):
    global N_H,N_W,N_C
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img,channels=N_C)
    img = tf.image.resize(img, [N_H, N_W],preserve_aspect_ratio=False)
    img = tf.cast(img,tf.float32)
    img = img*(2.0/255.0)-1.0
    return img

def convert_inv_images(img):
    return tf.image.flip_left_right(img)
