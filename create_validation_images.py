import data_processing as dp
from PIL import Image
import numpy as np
import os

def load_validation_images(dir_data_origin,fps_shape,num_fps,deterioration):
    dp.N_H = fps_shape[0]
    dp.N_W = fps_shape[1]
    dp.N_C = fps_shape[2]

    counter = 0
    for root,folders,files in os.walk(dir_data_origin):
        for file in files:
            file_dir = os.path.join(root,file)
            img_read = dp.read_orig_images(file_dir).numpy().reshape(1,dp.N_H,dp.N_W,dp.N_C)
            img_tar = np.concatenate((img_tar,img_read),0) if counter else img_read

            img_mod = np.copy(img_read).reshape(dp.N_H,dp.N_W,dp.N_C)
            img_mod = dp.dicc_map_funcs[deterioration](img_mod)
            img_mod = img_mod.reshape(1,dp.N_H,dp.N_W,dp.N_C)

            img_val = np.concatenate((img_val,img_mod),0) if counter else img_mod
            counter+=1
            if counter == num_fps:
                break
        if counter == num_fps:
            break

    return img_val/127.5-1.0,img_tar/127.5-1.0

def create_validation_images(model,dir_data_origin,execution_name,fps_shape,num_fps,deterioration):

    names_cats = ["to_enh","enhan","tar"]
    dir_ouput_data = os.path.join("./Output_valid",execution_name,"images")
    if not os.path.exists(dir_ouput_data): os.makedirs(dir_ouput_data)

    fps_to_enhance,fps_target = load_validation_images(dir_data_origin,fps_shape,num_fps,deterioration)
    fps_enhanced = model.generator(fps_to_enhance,training=False).numpy()

    if np.shape(fps_to_enhance)[3] == 1:
        fps_to_enhance = np.squeeze(fps_to_enhance,axis=3)
        fps_enhanced = np.squeeze(fps_enhanced,axis=3)
        fps_target = np.squeeze(fps_target,axis=3)

    for i in range(num_fps):
        for imgs,cat_name in zip([fps_to_enhance,fps_enhanced,fps_target],names_cats):
            img = Image.fromarray(((imgs[i,:]+1.0)*127.5).clip(0,255).astype(np.uint8))
            img.save(os.path.join(dir_ouput_data,"{}-{}.png".format(cat_name,i)))
