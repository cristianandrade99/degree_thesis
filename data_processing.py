import tensorflow as tf
from PIL import Image
import pathlib as pl
import numpy as np
import cv2
import os

import cris_utils as cu
import keys as km

img_validation_images_folder = "Img_Validation_images"

AUTOTUNE = tf.data.experimental.AUTOTUNE

N_H = 28
N_W = 28
N_C = 1

elipse_conf = []
blur_conf = []
func_keys = []

def load_process_fp_dataset(config):
    global N_H,N_W,N_C,elipse_conf,func_keys

    batch_size = config[km.batch_size_k]
    dir,patt = config[km.data_dir_patt_k][0],config[km.data_dir_patt_k][1]
    percent = config[km.data_percent_k]
    input_shape = config[km.fps_shape_k]
    func_keys = config[km.func_keys_k]
    run_desc = config[km.run_desc_k]

    N_H = input_shape[0]
    N_W = input_shape[1]
    N_C = input_shape[2]

    outputs_folder = cu.create_output_folders(run_desc)
    patterns,num_files = list_folder_patterns(dir,patt)
    num_data = int(num_files*percent/100)

    dataset = tf.data.Dataset.list_files(patterns,shuffle=True)\
    .shuffle(4096)\
    .take(num_data)\
    .map(read_orig_images,num_parallel_calls=AUTOTUNE)

    for key in func_keys:
        dataset = dataset.map(dicc_map_funcs_tf[key], num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size,True)\
    .cache("./{}/{}/cache.temp".format(outputs_folder,cu.cache_folder_name))\
    .prefetch(buffer_size=AUTOTUNE)

    return dataset

def load_verification_images(fps_shape,num_fps):
    global N_H,N_W,N_C,func_keys

    N_H = fps_shape[0]
    N_W = fps_shape[1]
    N_C = fps_shape[2]

    validation_images_source = "./{}".format(img_validation_images_folder)
    counter = 0
    for root,folders,files in os.walk(validation_images_source):
        for file in files:
            file_dir = os.path.join(root,file)
            img_read = read_orig_images(file_dir).numpy().reshape(1,N_H,N_W,N_C)
            img_tar = np.concatenate((img_tar,img_read),0) if counter else img_read

            img_mod = np.copy(img_read).reshape(N_H,N_W,N_C)
            for key in func_keys:
                img_mod = dicc_map_funcs[key](img_mod)
            img_mod = img_mod.reshape(1,N_H,N_W,N_C)

            img_val = np.concatenate((img_val,img_mod),0) if counter else img_mod
            counter+=1
            if counter == num_fps:
                break
        if counter == num_fps:
            break

    return img_val/127.5-1.0,img_tar/127.5-1.0

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

    np.random.shuffle(patterns)
    np.random.shuffle(patterns)

    return patterns,num_files

def read_orig_images(file_path):
    global N_H,N_W,N_C
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img,channels=N_C)
    img = tf.image.resize(img, [N_H, N_W],preserve_aspect_ratio=False)
    return img

# MAPPING METHODS
def tf_elipse(img):
    img_elip = tf.numpy_function(elipse,[img],tf.float32)
    return img_elip/127.5-1.0,img/127.5-1.0

def tf_blur(img):
    img_blur = tf.numpy_function(blur,[img],tf.float32)
    return img_blur/127.5-1.0,img/127.5-1.0

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

def blur(img):
    goal = 250
    sd=18

    mean_calc = np.mean(img)
    val = 9
    mod = val if goal > mean_calc else -val
    while  mean_calc <= goal-val/2 or mean_calc >= goal+val/2:
        img=(img+mod).clip(0,255)
        mean_calc = np.mean(img)

    val = 1
    mod = val if goal > mean_calc else -val
    while  mean_calc.astype(int)!=goal:
        img=(img+mod).clip(0,255)
        mean_calc = np.mean(img)

    img+=np.random.normal(0,sd,img.shape)

    return img

dicc_map_funcs_tf = {
    km.elip_image_k: tf_elipse,
    km.blur_image_k: tf_blur
}

dicc_map_funcs = {
    km.elip_image_k: elipse,
    km.blur_image_k: blur
}
