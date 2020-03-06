import tensorflow as tf
import numpy as np
import sys

import conv_deconv_blocks as cdb
import conv_deconv_models as md
import data_processing as dp
import cris_utils as cu

print("Versión de tensorflow: " + str(tf.__version__))
print("Comprobación de la GPU:",tf.config.experimental.list_physical_devices('GPU'),"\n")

run_description = sys.argv[1] if len(sys.argv)>1 else "DEFAULT"

N_H = 64
N_W = 64
N_C = 1

config = {
    md.batch_size_k: 16,
    md.latent_dim_k: 100,
    md.data_dir_patt_k: ["data_FVC2006","png"],
    md.fps_shape_k: (N_H,N_W,N_C)
}

ds_data_dirs = dp.load_process_fp_dataset(config[md.data_dir_patt_k],config[md.fps_shape_k],config[md.batch_size_k])

gan_cvae_train_conf = {
    md.num_epochs_k: 5,
    md.num_images_k: 10,
    md.checkpoints_frecuency_k: 10,
    md.use_latest_checkpoint_k: False,
    md.disc_learning_rate_k: 0.0002,
    md.gen_learning_rate_k: 0.0002,
    md.dataset_k: ds_data_dirs,
    md.num_histograms_k: 0
}

cvae_enc_convs = [(16,True,cdb.lr_act,3,2),
              (32,False,cdb.lr_act,3,2),
              (64,True,cdb.lr_act,3,2),
             (128,False,cdb.lr_act,3,2),
             (256,False,cdb.lr_act,3,2)]

cvae_dec_deconvs = [(512,True,cdb.r_act,3,2),
               (256,True,cdb.r_act,3,2),
               (128,True,cdb.r_act,3,2),
               (64,True,cdb.r_act,3,2),
               (N_C,False,cdb.th_act,3,2)]

gan_disc_convs = [(64,False,cdb.lr_act,3,2),
              (128,True,cdb.lr_act,3,2),
              (256,True,cdb.lr_act,3,2),
             (512,True,cdb.lr_act,3,2)]

n_cvae_conv_ls = len(cvae_enc_convs)
n_cvae_deconv_ls = len(cvae_dec_deconvs)

max_layers = cu.max_conv_deconv_layers(N_H)
assert (n_cvae_conv_ls<=max_layers and n_cvae_deconv_ls<=max_layers), "Incorrect number of layers for cvae model"

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

gan_cvae = md.GAN_CVAE(cvae_enc_config,cvae_dec_config,gan_disc_config,config,run_description)
gan_cvae.train(gan_cvae_train_conf)
