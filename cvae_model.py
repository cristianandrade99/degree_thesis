import conv_deconv_blocks as cdb
import matplotlib.pyplot as plt
import data_processing as dp
import custom_layers as cl
import tensorflow as tf
import tb_module as tb
import numpy as np
import time

# Dictionary keys
adam_alpha_k = "adam_alpha" #1e-4
checkpoints_folder_k = "checkpoints_folder"
max_checkpoints_k = "max_checkpoints" #2
dataset_k = "dataset"
use_latest_checkpoint_k = "use_latest_checkpoint"
num_epochs_k = "num_epochs_k"
percent_progress_savings_k = "percent_progress_savings"
num_images_k = "num_images"
batch_size_k = "batch_size"
create_checkpoints_k = "create_checkpoints"
latent_dim_k = "latent_dim"
types_losses_k = "types_losses"
alphas_losses_k = "alphas_losses"
tensorboard_folder_name_k = "tensorboard_folder_name"
data_dir_patt_k = "data_dir_patt"

# Losses
cross_loss = "Cross Entropy Loss"
square_loss = "Mean Square Error"
tv_loss = "Total Variation"
ssim_loss = "SSIM Loss"
kl_loss = "KL Loss"
total_loss_k = "Total Loss"

# Losses
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_squared_error = tf.keras.losses.MeanSquaredError()

class CVAE(tf.keras.Model):
    def __init__(self,enc_config,dec_config,config):
        super(CVAE, self).__init__()

        self.in_shape = enc_config[cdb.input_shape_k]

        self.encoder = cdb.encoder_module(None,enc_config)
        self.sampler = cl.Sample()
        self.decoder = cdb.decoder_module(None,dec_config)

        self.cvae_optimizer = tf.keras.optimizers.Adam(config[adam_alpha_k])

        self.tf_checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                                 sampler=self.sampler,
                                                 decoder=self.decoder,
                                                 cvae_optimizer=self.cvae_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.tf_checkpoint,config[checkpoints_folder_k],max_to_keep=config[max_checkpoints_k])

        self.batch_size = config[batch_size_k]
        self.latent_dim = config[latent_dim_k]

    @tf.function
    def train_step(self,images_batch,losses_tuple):

        actual_losses = {}
        total_loss = 0.0

        with tf.GradientTape() as model_tape:

            mean, logvar, images_batch_reconstructed = self.encode_decode_images(images_batch[0:1,:],True)
            losses,alphas = losses_tuple

            for loss,alp in zip(losses,alphas):
                act_loss = 0.0

                if loss == square_loss:
                    act_loss = alp*mean_squared_error(images_batch,images_batch_reconstructed)
                    actual_losses[square_loss] = act_loss

                elif loss == kl_loss:
                    act_loss = alp*tf.reduce_mean( 0.5*(tf.square(mean)+tf.exp(logvar)-1-logvar) )
                    actual_losses[kl_loss] = act_loss

                elif loss == ssim_loss:
                    act_loss = alp*tf.reduce_mean( tf.image.ssim(images_batch,images_batch_reconstructed,2.0) )
                    actual_losses[ssim_loss] = act_loss

                elif loss == cross_loss:
                    act_loss = alp*binary_cross_entropy(images_batch,images_batch_reconstructed)
                    actual_losses[cross_loss] = act_loss

                elif loss == tv_loss:
                    act_loss = alp*tf.reduce_mean(tf.image.total_variation(images_batch_reconstructed))
                    actual_losses[tv_loss] = act_loss

                total_loss += act_loss

        actual_losses[total_loss_k] = total_loss

        model_gradients = model_tape.gradient(total_loss,self.trainable_variables)
        self.cvae_optimizer.apply_gradients(zip(model_gradients,self.trainable_variables))

        return actual_losses

    def train(self,train_conf):

        print("training started")
        verification_Images,verif_pair_images = self.init_verif_pair_images()

        dataset = train_conf[dataset_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        num_epochs = train_conf[num_epochs_k]
        percent_progress_savings = train_conf[percent_progress_savings_k]
        num_images = train_conf[num_images_k]
        create_checkpoints = train_conf[create_checkpoints_k]
        losses_tuple = train_conf[types_losses_k],train_conf[alphas_losses_k]
        tensorboard_folder_name = train_conf[tensorboard_folder_name_k]

        train_summary_writer = tb.summary_writer(tensorboard_folder_name)

        if use_latest_checkpoint:
            self.tf_checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        epochs_progess_savings = [int((percent/100)*num_epochs) for percent in percent_progress_savings]

        for epoch_index in range(num_epochs):
            start_time = time.time()

            for images_batch in dataset:
                actual_losses = self.train_step(images_batch,losses_tuple)

            epoch_index_1 = epoch_index + 1

            self.log_training_tb(actual_losses,epoch_index_1,start_time,train_summary_writer)

            if epoch_index_1%int(num_epochs/num_images) == 0:
                _,_,ver_im_rec = self.encode_decode_images(verification_Images)
                for i in range(3):
                    verif_pair_images = tf.concat([verif_pair_images,verification_Images[i:i+1,:]],0)
                    verif_pair_images = tf.concat([verif_pair_images,ver_im_rec[i:i+1,:]],0)

            if create_checkpoints:
                if epoch_index_1 in epochs_progess_savings:
                    self.checkpoint_manager.save()

        with train_summary_writer.as_default():
            tf.summary.image("Images History",(verif_pair_images+1.0)/2.0,step=0,max_outputs=1000)

        print("training finished")

    def log_training_tb(self,actual_losses,epoch_index_1,start_time,train_summary_writer):
        with train_summary_writer.as_default():
            transc_time = np.round(time.time()-start_time,2)
            print("Epoch "+str(epoch_index_1)+": "+str(transc_time)+" s")
            tf.summary.scalar("Time per Epoch [s]",transc_time,step=epoch_index_1)
            for loss_k in actual_losses:
                tf.summary.scalar(loss_k, actual_losses[loss_k], step=epoch_index_1)
            tf.summary.scalar(total_loss_k, actual_losses[total_loss_k], step=epoch_index_1)

    def encode_decode_images(self,images,training=False):
        mean_logvar = self.encoder(images,training=training)
        mean, logvar =  tf.split(mean_logvar,2,1)
        z = self.sampler(mean,logvar)
        decoded_image = self.decoder(z,training=training)
        return mean, logvar, decoded_image

    def init_verif_pair_images(self):
        verification_Images = dp.load_verification_images(self.in_shape[0],self.in_shape[1],self.in_shape[2])[0:3,:]
        _,_,init_rec_images = self.encode_decode_images(verification_Images)
        verif_pair_images = verification_Images[0:1,:]
        verif_pair_images = tf.concat([verif_pair_images,init_rec_images[0:1,:]],0)
        verif_pair_images = tf.concat([verif_pair_images,verification_Images[1:2,:]],0)
        verif_pair_images = tf.concat([verif_pair_images,init_rec_images[1:2,:]],0)
        verif_pair_images = tf.concat([verif_pair_images,verification_Images[2:3,:]],0)
        verif_pair_images = tf.concat([verif_pair_images,init_rec_images[2:3,:]],0)
        return verification_Images,verif_pair_images
