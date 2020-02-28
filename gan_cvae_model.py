import conv_deconv_blocks as cdb
import matplotlib.pyplot as plt
import data_processing as dp
import custom_layers as cl
import cris_utils as cu
import tensorflow as tf
import numpy as np
import warnings
import time

#warnings.simplefilter('error', UserWarning)

fps_shape_k = cdb.fps_shape_k
cvae_checkpoints_folder_k = "cvae_checkpoints_folder"
gan_checkpoints_folder_k = "gan_checkpoints_folder"
max_checkpoints_k = "max_checkpoints"
latent_dim_k = "latent_dim"
batch_size_k = "batch_size"
data_dir_patt_k = "data_dir_patt"

dataset_k = "dataset"
use_latest_checkpoint_k = "use_latest_checkpoint"
num_epochs_k = "num_epochs_k"
checkpoints_frecuency_k = "checkpoints_frecuency"
num_images_k = "num_images"
types_losses_k = "types_losses"
alphas_losses_k = "alphas_losses_k"
tensorboard_folder_k = "tensorboard_folder_k"
out_images_folder_k = "out_images_folder"

cvae_model = "cvae_model"
gan_model = "gan_model"

square_loss = "Mean Squared Error"
kl_loss = "KL Divergence"
ssim_loss = "SSIM Loss"
tv_loss = "Total Variation Loss"
cross_loss = "CrossEntropy Loss"
total_loss_k = "Total Loss"

gan_gen_loss_k = "Gan Generator Loss"
gan_disc_loss_k = "Gan Discriminator Loss"

binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class GAN_CVAE(tf.keras.Model):
    def __init__(self,cvae_enc_config,cvae_dec_config,disc_config,config):
        super(GAN_CVAE,self).__init__()

        self.fps_shape = config[fps_shape_k]

        self.cvae_encoder = cdb.encoder_module(cvae_enc_config)
        self.cvae_sampler = cl.Sample()
        self.cvae_decoder = cdb.decoder_module(cvae_dec_config)
        self.cvae_optimizer = tf.keras.optimizers.Adam()
        self.cvae_checkpoint = tf.train.Checkpoint(cvae_encoder=self.cvae_encoder,
                                                   cvae_sampler=self.cvae_sampler,
                                                   cvae_decoder=self.cvae_decoder,
                                                   cvae_optimizer=self.cvae_optimizer)
        self.cvae_checkpoint_manager = tf.train.CheckpointManager(self.cvae_checkpoint,
                                                                  config[cvae_checkpoints_folder_k],
                                                                  max_to_keep=config[max_checkpoints_k])

        self.gan_discriminator = cdb.encoder_module(disc_config)
        self.gan_gen_optimizer = tf.keras.optimizers.Adam()
        self.gan_disc_optimizer = tf.keras.optimizers.Adam()
        self.gan_checkpoint = tf.train.Checkpoint(gan_discriminator=self.gan_discriminator,
                                                   gan_gen_optimizer=self.gan_gen_optimizer,
                                                   gan_disc_optimizer=self.gan_disc_optimizer)
        self.gan_checkpoint_manager = tf.train.CheckpointManager(self.gan_checkpoint,
                                                                  config[gan_checkpoints_folder_k],
                                                                  max_to_keep=config[max_checkpoints_k])

    # CVAE METHODS
    @tf.function
    def cvae_train_step(self,fps_batch,losses_tuple):
        losses,alphas = losses_tuple
        actual_losses = {}
        total_loss = 0.0

        with tf.GradientTape() as cvae_tape:
            mean,logvar,fps_processed = self.cvae_encode_decode_fps(fps_batch,True)

            for loss,alph in zip(losses,alphas):
                if loss == square_loss:
                    act_loss = alph*tf.reduce_mean(tf.keras.losses.MSE(fps_batch,fps_processed))
                    actual_losses[square_loss] = act_loss
                elif loss == kl_loss:
                    act_loss = alph*tf.reduce_mean(0.5*( tf.square(mean)+tf.exp(logvar)-1-logvar ))
                    actual_losses[kl_loss] = act_loss
                elif loss == ssim_loss:
                    act_loss = alph*tf.reduce_mean( tf.image.ssim(fps_batch,fps_processed,max_val=2.0) )
                    actual_losses[ssim_loss] = act_loss
                elif loss == tv_loss:
                    act_loss = alph*tf.reduce_mean( tf.image.total_variation(fps_processed) )
                    actual_losses[tv_loss] = act_loss
                elif loss == cross_loss:
                    act_loss = alph*binary_crossentropy(fps_batch,fps_processed)
                    actual_losses[cross_loss] = act_loss

                total_loss += act_loss
            actual_losses[total_loss_k] = total_loss

        cvae_trainable_variables = self.cvae_encoder.trainable_variables+self.cvae_decoder.trainable_variables
        cvae_gradients = cvae_tape.gradient(total_loss,cvae_trainable_variables)
        self.cvae_optimizer.apply_gradients(zip(cvae_gradients,cvae_trainable_variables))
        return actual_losses

    def cvae_train(self,train_conf):
        print("CVAE training started")
        start_time = time.time()

        dataset = train_conf[dataset_k]
        num_epochs = train_conf[num_epochs_k]
        losses_tuple = train_conf[types_losses_k],train_conf[alphas_losses_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        checkpoints_frecuency = train_conf[checkpoints_frecuency_k]
        num_images = train_conf[num_images_k]
        out_images_folder = train_conf[out_images_folder_k]

        tensorboard_folder = train_conf[tensorboard_folder_k]
        tf_summary_writer = cu.tf_summary_writer(tensorboard_folder)

        if use_latest_checkpoint:
            self.cvae_checkpoint.restore(self.cvae_checkpoint_manager.latest_checkpoint)

        for epoch_index in range(num_epochs):
            for fps_batch in dataset:
                actual_losses = self.cvae_train_step(fps_batch,losses_tuple)

            self.cvae_data_to_tensorboard(actual_losses,epoch_index,tf_summary_writer)
            self.progress_fps_to_folder(epoch_index+1,num_images,num_epochs,out_images_folder,4)
            self.cvae_save_checkpoint(epoch_index+1,checkpoints_frecuency)

        self.log_training_end(start_time,num_epochs)

    def cvae_encode_decode_fps(self,fps_batch,is_training=False):
        mean_logvar = self.cvae_encoder(fps_batch,training=is_training)
        mean,logvar = tf.split(mean_logvar,2,1)
        z = self.cvae_sampler(mean,logvar)
        fps_processed = self.cvae_decoder(z,training=is_training)
        return mean,logvar,fps_processed

    def cvae_data_to_tensorboard(self,actual_losses,epoch_index,tf_summary_writer):
        with tf_summary_writer.as_default():
            for loss_k in actual_losses:
                tf.summary.scalar(loss_k,actual_losses[loss_k],step=epoch_index)
            tf.summary.scalar(total_loss_k,actual_losses[total_loss_k],step=epoch_index)

    def cvae_save_checkpoint(self,epoch_index,checkpoints_frecuency):
        if( epoch_index%checkpoints_frecuency == 0 ):
            self.cvae_checkpoint_manager.save()

    def log_training_end(self,start_time,num_epochs):
        print("Training total time: "+str(np.round(time.time()-start_time,2)))
        print("Average time per epoch: "+str(np.round((time.time()-start_time)/num_epochs,2)))
        print("Training finished")

    # GAN METHODS
    @tf.function
    def gan_train_step(self,fps_batch):
        actual_losses = {}

        with tf.GradientTape() as gan_gen_tape, tf.GradientTape() as gan_disc_tape:
            _,_,fps_processed = self.cvae_encode_decode_fps(fps_batch,True)

            fps_batch_logits = self.gan_discriminator(fps_batch,training=True)
            fps_processed_logits = self.gan_discriminator(fps_processed,training=True)

            gen_loss = binary_crossentropy(tf.ones_like(fps_processed_logits),fps_processed_logits)
            disc_loss = binary_crossentropy(tf.ones_like(fps_batch_logits),fps_batch_logits)
            disc_loss += binary_crossentropy(tf.zeros_like(fps_processed_logits),fps_processed_logits)

            actual_losses[gan_gen_loss_k] = gen_loss
            actual_losses[gan_disc_loss_k] = disc_loss

        cvae_trainable_variables = self.cvae_encoder.trainable_variables+self.cvae_decoder.trainable_variables
        gen_gradients = gan_gen_tape.gradient(gen_loss,cvae_trainable_variables)
        disc_gradients = gan_disc_tape.gradient(disc_loss,self.gan_discriminator.trainable_variables)

        self.gan_gen_optimizer.apply_gradients(zip(gen_gradients,cvae_trainable_variables))
        self.gan_disc_optimizer.apply_gradients(zip(disc_gradients,self.gan_discriminator.trainable_variables))
        return actual_losses

    def gan_train(self,train_conf):
        print("GAN training started")
        start_time = time.time()

        dataset = train_conf[dataset_k]
        num_epochs = train_conf[num_epochs_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        checkpoints_frecuency = train_conf[checkpoints_frecuency_k]
        num_images = train_conf[num_images_k]
        out_images_folder = train_conf[out_images_folder_k]

        tensorboard_folder = train_conf[tensorboard_folder_k]
        tf_summary_writer = cu.tf_summary_writer(tensorboard_folder)

        if use_latest_checkpoint:
            self.gan_checkpoint.restore(self.gan_checkpoint_manager.latest_checkpoint)

        for epoch_index in range(num_epochs):
            for fps_batch in dataset:
                actual_losses = self.gan_train_step(fps_batch)

            self.gan_data_to_tensorboard(actual_losses,epoch_index,tf_summary_writer)
            self.progress_fps_to_folder(epoch_index+1,num_images,num_epochs,out_images_folder,4)
            self.gan_save_checkpoint(epoch_index+1,checkpoints_frecuency)

        self.log_training_end(start_time,num_epochs)

    def gan_data_to_tensorboard(self,actual_losses,epoch_index,tf_summary_writer):
        with tf_summary_writer.as_default():
            tf.summary.scalar(gan_gen_loss_k,actual_losses[gan_gen_loss_k],step=epoch_index)
            tf.summary.scalar(gan_disc_loss_k,actual_losses[gan_disc_loss_k],step=epoch_index)

    def gan_save_checkpoint(self,epoch_index,checkpoints_frecuency):
        if( epoch_index%checkpoints_frecuency == 0 ):
            self.gan_checkpoint_manager.save()

    # GENERAL METHODS
    def progress_fps_to_folder(self,epoch_index,num_images,num_epochs,out_images_folder,num_fps):

        if( epoch_index%(num_epochs/num_images) == 0 ):
            fps_batch = dp.load_verification_images(self.fps_shape,num_fps).numpy()
            _,_,fps_processed = self.cvae_encode_decode_fps(fps_batch)
            fps_processed = fps_processed.numpy()

            fps_batch_shapes = np.shape(fps_batch)
            if fps_batch_shapes[3] == 1:
                fps_batch = np.reshape(fps_batch,(fps_batch_shapes[0],fps_batch_shapes[1],fps_batch_shapes[2]))

            fps_processed_shapes = np.shape(fps_processed)
            if fps_processed_shapes[3] == 1:
                fps_processed = np.reshape(fps_processed,(fps_processed_shapes[0],fps_processed_shapes[1],fps_processed_shapes[2]))

            fig,axs = plt.subplots(num_fps,2,figsize=(5,5),constrained_layout=True)
            fig.suptitle("Epoch: {}".format(epoch_index))

            for i in range(num_fps):
                axs[i,0].imshow(fps_batch[i,:],cmap="gray")
                axs[i,1].imshow(fps_processed[i,:],cmap="gray")
                axs[i,0].axis('off')
                axs[i,1].axis('off')

            plt.savefig(out_images_folder+"/fp_at_epoch_{}.png".format(epoch_index),bbox_inches='tight')
            plt.close(fig)
