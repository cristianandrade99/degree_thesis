import keys as km
import json
import sys
import os

# DETERIORATION
# elip_image_k - blur_image_k
data_config = {
    km.fps_shape_k: (256,256,1),
    km.data_dir_patt_k: ["./Data/Olimpia","jpg"],
    km.num_images_training_k: 4,
    km.deter_func_key_k: km.elip_image_k,
    km.batch_size_k: 2
}

train_conf = {
    km.num_epochs_k: 2,
    km.gen_losses_k: [km.l1_loss],
    km.gen_alphas_losses_k: [50.0],
    km.alpha_ones_p_k: 0.9,
    km.gen_disc_loss_alphas_k: [1.0,0.5],
    km.gen_adam_params_k: [0.00018,0.5],
    km.disc_adam_params_k: [0.00018,0.5],
    km.total_epochs_to_save_imgs_k: 2,
    km.epochs_to_save_chkps_k: [1]
}

# GENERATOR AND DISCRIMINATOR
# p2p_paper_gen_conf_K
# p2p_paper_disc_conf_k
# p2p_lenovo_gen_conf_k
# p2p_lenovo_disc_conf_k
gen_disc_conf = {
    km.generator_arch_k: km.p2p_lenovo_gen_conf_k,
    km.discriminator_k: km.p2p_lenovo_disc_conf_k
}

overall_conf = {
    km.data_config_k: data_config,
    km.train_conf_k: train_conf,
    km.gen_disc_conf_k: gen_disc_conf
}

args,len_args = sys.argv,len(sys.argv)
if len_args == 2:
    file = open("./{}.json".format(args[1]),'w')
    json.dump(overall_conf,file)
    file.close()
else:
    print("Please specify execution name")
