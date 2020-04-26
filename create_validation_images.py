import sys
sys.path.append("C:/Users/Y50/Documents/Universidad/Tesis/Codigo")

import conv_deconv_blocks as cdb
import conv_deconv_models as md
import data_processing as dp
import tensorflow as tf
from PIL import Image
import numpy as np
import shutil
import os

def load_validation_images(source,fps_shape,num_fps,preprocess):
    dp.N_H = fps_shape[0]
    dp.N_W = fps_shape[1]
    dp.N_C = fps_shape[2]

    counter = 0
    for root,folders,files in os.walk(source):
        for file in files:
            file_dir = os.path.join(root,file)
            img_read = dp.read_orig_images(file_dir).numpy().reshape(1,dp.N_H,dp.N_W,dp.N_C)
            img_tar = np.concatenate((img_tar,img_read),0) if counter else img_read

            img_mod = np.copy(img_read).reshape(dp.N_H,dp.N_W,dp.N_C)
            img_mod = dp.dicc_map_funcs[preprocess](img_mod)
            img_mod = img_mod.reshape(1,dp.N_H,dp.N_W,dp.N_C)

            img_val = np.concatenate((img_val,img_mod),0) if counter else img_mod
            counter+=1
            if counter == num_fps:
                break
        if counter == num_fps:
            break

    return img_val/127.5-1.0,img_tar/127.5-1.0

def create_validation_images(model,dir_sources,source,dir_verif_imgs,folder,fps_shape,num_fps,preprocess):

    names_cats = ["to_enh","enhan","tar"]
    dir_images_verif = os.path.join(dir_verif_imgs,"{}_{}_{}".format(folder,preprocess,source.replace("/","-")),"images")
    if not os.path.exists(dir_images_verif): os.makedirs(dir_images_verif)

    fps_to_enhance,fps_target = load_validation_images(dir_sources,fps_shape,num_fps,preprocess)
    fps_enhanced = model.generator(fps_to_enhance,training=False).numpy()

    if np.shape(fps_to_enhance)[3] == 1:
        fps_to_enhance = np.squeeze(fps_to_enhance,axis=3)
        fps_enhanced = np.squeeze(fps_enhanced,axis=3)
        fps_target = np.squeeze(fps_target,axis=3)

    for i in range(num_fps):
        for imgs,cat_name in zip([fps_to_enhance,fps_enhanced,fps_target],names_cats):
            img = Image.fromarray(((imgs[i,:]+1.0)*127.5).clip(0,255).astype(np.uint8))
            img.save(os.path.join(dir_images_verif,"{}-{}.png".format(cat_name,i)))

def remove_folders(dir_folders):
    for folder in os.listdir(dir_folders):
        dir_to_remove = os.path.join(dir_folders,folder)
        shutil.rmtree(dir_to_remove)

def run(processes,sources,num_fps):

    dp.elipse_conf = [15,1,2,20,30,25,3,180,85]
    config = {md.fps_shape_k: (256,256,1)}
    dir_tesis_codigo = "C:/Users/Y50/Documents/Universidad/Tesis/Codigo"
    dir_sources = os.path.join(dir_tesis_codigo,"Data","JAVIER_REC")
    dir_verif_imgs = os.path.join(dir_tesis_codigo,"Output_data_validation")
    remove_folders(dir_verif_imgs)
    #generator,discriminator = cdb.create_gen_disc(config)
    generator,discriminator = cdb.create_gen_disc_lenovo(config)
    dir_model_outputs = os.path.join(dir_tesis_codigo,"Output_data")
    preprocess = [dp.blur_image_k,dp.make_elipses_k]

    for folder in os.listdir(dir_model_outputs):
        dir_act_checkpoints_folder = os.path.join(dir_model_outputs,folder,"tf_Checkpoints")

        for preprocess in processes:
            for source in sources:

                p2p_model = md.P2P(generator,discriminator,None,"Data Gen")
                p2p_model.checkpoint = tf.train.Checkpoint(generator=p2p_model.generator,discriminator=p2p_model.discriminator,gen_optimizer=p2p_model.gen_optimizer,disc_optimizer=p2p_model.disc_optimizer)
                p2p_model.checkpoint_manager = tf.train.CheckpointManager(p2p_model.checkpoint,dir_act_checkpoints_folder,max_to_keep=md.max_checkpoints_to_keep)
                print(p2p_model.checkpoint_manager.latest_checkpoint)
                p2p_model.checkpoint.restore(p2p_model.checkpoint_manager.latest_checkpoint)
                create_validation_images(p2p_model,dir_sources,source,dir_verif_imgs,folder,config[md.fps_shape_k],num_fps,preprocess)

processes = [dp.make_elipses_k,dp.blur_image_k]
sources = ["Alive"]
num_fps = 40

run(processes,sources,num_fps)
