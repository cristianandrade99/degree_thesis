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
    md.data_dir_patt_k: ["./data_FVC2006","png"],
    md.fps_shape_k: (N_H,N_W,N_C)
}

ds_data_dirs = dp.load_process_fp_dataset(config[md.data_dir_patt_k],config[md.fps_shape_k],config[md.batch_size_k])

gan_train_conf = {
    md.num_epochs_k: 5,
    md.num_images_k: 10,
    md.checkpoints_frecuency_k: 10,
    md.use_latest_checkpoint_k: False,
    md.disc_learning_rate_k: 0.0002,
    md.gen_learning_rate_k: 0.0002,
    md.dataset_k: ds_data_dirs,
    md.num_histograms_k: 2
}

gan_gen_deconvs = [(512,True,cdb.r_act,3,2),
               (256,True,cdb.r_act,3,2),
               (128,True,cdb.r_act,3,2),
               (64,True,cdb.r_act,3,2),
               (N_C,False,cdb.th_act,3,2)]

gan_disc_convs = [(64,False,cdb.lr_act,3,2),
              (128,True,cdb.lr_act,3,2),
              (256,True,cdb.lr_act,3,2),
             (512,True,cdb.lr_act,3,2)]

n_gan_gen_deconv_ls = len(gan_gen_deconvs)
n_gan_conv_ls = len(gan_disc_convs)

max_layers = cu.max_conv_deconv_layers(N_H)
assert (n_gan_conv_ls<=max_layers and n_gan_gen_deconv_ls<=max_layers), "Incorrect number of layers for cvae model"

hw_f_v = N_H/np.power(2,n_gan_gen_deconv_ls)
gan_gen_config = {
    cdb.dec_den_info_k: ([hw_f_v,hw_f_v,256],False,cdb.r_act,(config[md.latent_dim_k],)),
    cdb.enc_dec_lys_info_k: gan_gen_deconvs,
}

gan_disc_config = {
    cdb.fps_shape_k: config[md.fps_shape_k],
    cdb.enc_dec_lys_info_k: gan_disc_convs,
    cdb.enc_fin_den_len_k: 1
}

gan = md.GAN(gan_gen_config,gan_disc_config,config,run_description)
gan.train(gan_train_conf)
