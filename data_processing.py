import tensorflow as tf
from PIL import Image
import pathlib as pl
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import conv_deconv_models as md
import cris_utils as cu

mirror_data_k = "mirror_data"
make_elipses_k = "make_elipses"
elipse_conf_k = "elipse_conf"

AUTOTUNE = tf.data.experimental.AUTOTUNE

N_H = 28
N_W = 28
N_C = 1
N_C_loaded = 1
elipse_conf = []

def load_process_fp_dataset(config):
    global N_H,N_W,N_C,elipse_conf

    dir,patt = config[md.data_dir_patt_k][0],config[md.data_dir_patt_k][1]
    patterns,num_files = list_folder_patterns(dir,patt)
    ds_data_dirs_orig = tf.data.Dataset.list_files(patterns,shuffle=True)

    '''data_list = list(ds_data_dirs_orig.as_numpy_iterator())
    print("")
    print("Fingerprints Loaded:",len(data_list),"from:",dir,"\n")'''

    input_shape = config[md.fps_shape_k]
    N_H = input_shape[0]
    N_W = input_shape[1]
    N_C = input_shape[2]
    elipse_conf = config[elipse_conf_k]

    ds_data_dirs_orig = ds_data_dirs_orig.map(read_orig_images, num_parallel_calls=AUTOTUNE)
    ds_data_dirs_inv = ds_data_dirs_orig.map(convert_inv_images, num_parallel_calls=AUTOTUNE)

    ds_data_dirs = ds_data_dirs_orig

    if config[mirror_data_k]:
        ds_data_dirs = ds_data_dirs.concatenate(ds_data_dirs_inv)

    if config[make_elipses_k]:
        ds_data_dirs = ds_data_dirs.map(elipse_tf, num_parallel_calls=AUTOTUNE)

    ds_data_dirs = ds_data_dirs.shuffle(buffer_size=4096,reshuffle_each_iteration=True)
    ds_data_dirs = ds_data_dirs.batch(config[md.batch_size_k],True)
    ds_data_dirs = ds_data_dirs.prefetch(buffer_size=AUTOTUNE)

    '''data_list_batched = list(ds_data_dirs.as_numpy_iterator())
    print("Batches Created:",len(data_list_batched))
    print("shape of batch:",data_list_batched[0].shape,"\n")'''

    return ds_data_dirs,num_files

def load_verification_images(fps_shape,num_fps):
    global N_H,N_W,N_C

    N_H = fps_shape[0]
    N_W = fps_shape[1]
    N_C = fps_shape[2]

    validation_images_source = "./Img_Validation_images"

    counter = 0
    for root,folders,files in os.walk(validation_images_source):
        for file in files:
            file_dir = os.path.join(root,file)
            img = read_orig_images(file_dir).numpy()
            img = elipse(img).reshape(1,N_H,N_W,N_C)
            img_val = np.concatenate((img_val,img),0) if counter else img

            counter+=1
            if counter == num_fps:
                break

        if counter == num_fps:
            break

    return img_val/127.5-1

def list_folder_patterns(root_dir,patern):
    patterns = []
    num_files = 0
    for root,folders,files in os.walk(root_dir):
        num_files+=len(files)
        for file in files:
            file_dir = str(os.path.join(root,file))
            parent = str(pl.Path(file_dir).parent)
            act_patern = "./"+(str(parent)+"/*"+patern)
            if act_patern not in patterns:
                patterns.append(act_patern)
                continue
    return patterns,num_files

def read_orig_images(file_path):
    global N_H,N_W,N_C
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img,channels=N_C)
    img = tf.image.resize(img, [N_H, N_W],preserve_aspect_ratio=False)
    img = tf.cast(img,tf.float32)
    return img

def convert_inv_images(img):
    return tf.image.flip_left_right(img)

def elipse_tf(img):
    img_elip = tf.numpy_function(elipse,[img],tf.float32)
    return img_elip/127.5-1,img/127.5-1

def elipse(img):

    n_holes = int(cu.calc_unit_to_range(np.random.rand(),elipse_conf[1],elipse_conf[2]))
    rand = np.random.rand(n_holes,4)
    pos = cu.calc_unit_to_range(rand[:,0:2],elipse_conf[0]+2*elipse_conf[4],N_H-elipse_conf[0]-2*elipse_conf[4]).astype(int)
    l_shorts = cu.calc_unit_to_range(rand[:,2],elipse_conf[3],elipse_conf[4]).astype(int)
    min_percent = 1.25
    percents = cu.calc_unit_to_range(rand[:,3],min_percent,min_percent + (2-min_percent)*(elipse_conf[5]/100) )
    l_longs = (l_shorts*percents).astype(int)

    for k in range(n_holes):
        x0,y0= pos[k,0],pos[k,1]

        a,b = l_shorts[k],l_longs[k]
        if np.random.rand() < 0.2:
            a,b = l_longs[k],l_shorts[k]

        rectangle = np.copy(img[y0-b:y0+b,x0-a:x0+a,0])
        filtered = cv2.filter2D(rectangle,-1,np.ones((elipse_conf[6],elipse_conf[6]))/(elipse_conf[6]**2))
        gauss = np.random.normal(elipse_conf[7],elipse_conf[8],rectangle.shape).astype(int)

        img[y0-b:y0+b,x0-a:x0+a,0] = (filtered+gauss).clip(0,255)

        a2 = a**2
        b2 = b**2
        a2b2 = a2*b2

        for i in range(y0-b,y0+b):
            for j in range(x0-a,x0+a):
                if i >= 0 and i < N_H and j >= 0 and j < N_W:
                    if not ( (b2)*((j-x0)**2) +(a2)*((i-y0)**2) <= a2b2 ):
                        img[i,j,0]=rectangle[i-y0+b,j-x0+a]
    return img
