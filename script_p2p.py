# DO NOT DELETE - PATH FOR TF IN CLUSTER
import path_cluster

import pix2pix_configurations as p2p_c
import data_processing as dp
import pix2pix as p2p
import keys as km
import sys

run_description = sys.argv[1] if len(sys.argv)>1 else "DEFAULT"

dp.elipse_conf = [15,1,2,20,30,25,3,180,85]

# blur_image_k - elip_image_k
general_config = {
    km.batch_size_k: 2,
    km.data_dir_patt_k: ["./Data/JAVIER_REC","png"],
    km.data_percent_k: 0.01,
    km.fps_shape_k: (256,256,1),
    km.func_keys_k: [km.blur_image_k],
    km.model_k: "P2P",
    km.run_desc_k: run_description
}

gen_disc_config = p2p_c.lenovo_configuration()
#gen_disc_config = p2p_c.standard_configuration()
generator_discriminator = p2p_c.create_standard_pix2pix(gen_disc_config,general_config[km.fps_shape_k])

data_info = dp.load_process_fp_dataset(general_config)

train_conf = {
    km.num_epochs_k: 2,
    km.num_images_k: 2,
    km.epochs_to_save_k: [0],
    km.use_latest_checkpoint_k: False,
    km.types_losses_k: [km.l1_loss],
    km.alphas_losses_k: [50.0],
    km.gen_adam_params_k: [0.00018,0.5],
    km.disc_adam_params_k: [0.00028,0.5],
    km.gen_disc_loss_alphas_k: [1.0,0.5],
    km.alpha_ones_p_k: 0.9,
    km.num_histograms_k: 0,
    km.data_info_k: data_info,
    km.num_progress_images_k: 8
}

model = p2p.Pix2Pix(general_config,gen_disc_config)
model.update_gen_disc(generator_discriminator)
model.train(train_conf)
