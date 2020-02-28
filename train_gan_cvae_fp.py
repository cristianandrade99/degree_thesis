import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

import conv_deconv_blocks as cdb
import data_processing as dp
import gan_cvae_model as md
import cris_utils as cu

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(8)

print("\n","Versión de tensorflow: " + str(tf.__version__),"\n")
print("Comprobación de la GPU:",tf.config.experimental.list_physical_devices('GPU'),"\n")

N_H = 128
N_W = 128
N_C = 1

cvae_enc_convs = [(16,False,cdb.lr_act,3,2),
              (32,False,cdb.lr_act,3,2),
              (64,False,cdb.lr_act,3,2),
             (128,False,cdb.lr_act,3,2),
             (256,False,cdb.lr_act,3,2)]

cvae_dec_deconvs = [(256,False,cdb.r_act,3,2),
               (128,False,cdb.r_act,3,2),
               (64,False,cdb.r_act,3,2),
               (32,False,cdb.r_act,3,2),
               (N_C,False,cdb.th_act,3,2)]

gan_disc_convs = [(16,False,cdb.lr_act,3,2),
              (32,False,cdb.lr_act,3,2),
              (64,False,cdb.lr_act,3,2),
             (128,False,cdb.lr_act,3,2),
             (256,False,cdb.lr_act,3,2)]

n_cvae_conv_ls = len(cvae_enc_convs)
n_cvae_deconv_ls = len(cvae_dec_deconvs)

max_layers = cu.max_conv_deconv_layers(N_H)
assert (n_cvae_conv_ls<=max_layers and n_cvae_deconv_ls<=max_layers), "Incorrect number of layers for cvae model"

config = {
    md.batch_size_k: 16,
    md.latent_dim_k: 32,
    md.data_dir_patt_k: ["data_FVC2006","png"],
    md.fps_shape_k: (N_H,N_W,N_C),
    md.cvae_checkpoints_folder_k: "./tf_Checkpoints/CVAE",
    md.gan_checkpoints_folder_k: "./tf_Checkpoints/GAN",
    md.max_checkpoints_k: 2
}

cvae_enc_config = {
    cdb.fps_shape_k: config[md.fps_shape_k],
    cdb.enc_dec_lys_info_k: cvae_enc_convs,
    cdb.enc_fin_den_len_k: 2*config[md.latent_dim_k]
}

hw_f_v = N_H/np.power(2,n_cvae_deconv_ls)

cvae_dec_config = {
    cdb.dec_den_info_k: ([hw_f_v,hw_f_v,256],False,cdb.r_act,(config[md.latent_dim_k],)),
    cdb.enc_dec_lys_info_k: cvae_dec_deconvs,
}

gan_disc_config = {
    cdb.fps_shape_k: config[md.fps_shape_k],
    cdb.enc_dec_lys_info_k: gan_disc_convs,
    cdb.enc_fin_den_len_k: 1
}

ds_data_dirs = dp.load_process_fp_dataset(config[md.data_dir_patt_k],config[md.fps_shape_k],config[md.batch_size_k])

gan_cvae = md.GAN_CVAE(cvae_enc_config,cvae_dec_config,gan_disc_config,config)

cvae_train_conf = {
    md.num_epochs_k: 5,
    md.num_images_k: 10,
    md.checkpoints_frecuency_k: 10,
    md.use_latest_checkpoint_k: False,
    md.types_losses_k: [md.square_loss,md.kl_loss,md.ssim_loss],
    md.alphas_losses_k: [1.0,0.01,0.5],
    md.dataset_k: ds_data_dirs,
    md.tensorboard_folder_k: "./tf_Tensorboard_logs/CVAE",
    md.out_images_folder_k: "./img_Performance_Images/CVAE",
}

gan_train_conf = {
    md.num_epochs_k: 5,
    md.num_images_k: 5,
    md.checkpoints_frecuency_k: 5,
    md.use_latest_checkpoint_k: False,
    md.dataset_k: ds_data_dirs,
    md.tensorboard_folder_k: "./tf_Tensorboard_logs/GAN",
    md.out_images_folder_k: "./img_Performance_Images/GAN",
}

gan_cvae.cvae_train(cvae_train_conf)
gan_cvae.gan_train(gan_train_conf)
