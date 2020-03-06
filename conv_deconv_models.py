import conv_deconv_blocks as cdb
import matplotlib.pyplot as plt
import data_processing as dp
import custom_layers as cl
import cris_utils as cu
import tensorflow as tf
import datetime as dt
import numpy as np
import warnings
import time

fps_shape_k = cdb.fps_shape_k
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

learning_rate_k = "learning_rate"
disc_learning_rate_k = "disc_learning_rate"
gen_learning_rate_k = "gen_learning_rate"

cvae_model = "cvae_model"
gan_model = "gan_model"

square_loss = "Mean Squared Error"
kl_loss = "KL Divergence"
ssim_loss = "SSIM Loss"
tv_loss = "Total Variation Loss"
cross_loss = "CrossEntropy Loss"
total_loss_k = "Total Loss"

gan_disc_batch_loss_k = "Gan Discriminator Batch Loss"
gan_disc_processed_loss_k = "Gan Discriminator Processed Loss"
gan_disc_generated_loss_k = "Gan Discriminator Generated Loss"
gan_gen_loss_k = "Gan Generator Loss"

gan_disc_batch_accuracy_k = "Gan Discriminator Batch Accuracy"
gan_disc_processed_accurac_k = "Gan Discriminator Processed Accuracy"

disc_batch_gradients_k = "Discriminator Batch Gradients"
disc_processed_gradients_k = "Discriminator Processed Gradients"
gen_gradients_k = "Generator Gradients"

binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
max_checkpoints_to_keep = 2
histogram_frecuency = 2
num_progress_images = 4

class CVAE():
    def __init__(self,enc_config,dec_config,config,run_description):
        super(CVAE,self).__init__()

        self.run_description = run_description

        self.fps_shape = config[fps_shape_k]
        self.batch_size = config[batch_size_k]

        self.encoder = cdb.encoder_module(enc_config)
        self.sampler = cl.Sample()
        self.decoder = cdb.decoder_module(dec_config)

        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(self,fps_batch,losses_tuple):
        losses,alphas = losses_tuple
        actual_losses = {}
        total_loss = 0.0

        with tf.GradientTape() as tape:
            mean,logvar,fps_processed = self.encode_decode_fps(fps_batch,True)

            for loss,alph in zip(losses,alphas):
                if loss == square_loss:
                    actual_losses[square_loss] = alph*tf.reduce_mean(tf.keras.losses.MSE(fps_batch,fps_processed))
                elif loss == kl_loss:
                    actual_losses[kl_loss] = alph*tf.reduce_mean(0.5*( tf.square(mean)+tf.exp(logvar)-1-logvar ))
                elif loss == ssim_loss:
                    actual_losses[ssim_loss] = alph*tf.reduce_mean( tf.image.ssim(fps_batch,fps_processed,max_val=2.0) )
                elif loss == tv_loss:
                    actual_losses[tv_loss] = alph*tf.reduce_mean( tf.image.total_variation(fps_processed) )
                elif loss == cross_loss:
                    actual_losses[cross_loss] = alph*binary_crossentropy(fps_batch,fps_processed)

                total_loss += actual_losses[loss]

            actual_losses[total_loss_k] = total_loss

        trainable_variables = self.encoder.trainable_variables+self.decoder.trainable_variables
        gradients = tape.gradient(total_loss,trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,trainable_variables))
        return actual_losses,gradients

    def train(self,train_conf):
        print("CVAE training started")
        start_time = time.time()

        dataset = train_conf[dataset_k]
        num_epochs = train_conf[num_epochs_k]
        losses_tuple = train_conf[types_losses_k],train_conf[alphas_losses_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        checkpoints_frecuency = train_conf[checkpoints_frecuency_k]
        num_images = train_conf[num_images_k]
        learning_rate = train_conf[learning_rate_k]

        self.optimizer.learning_rate = learning_rate

        outputs_folder = cu.create_output_folders("CVAE",self.run_description)
        tf_summary_writer = cu.tf_summary_writer(outputs_folder)

        self.create_checkpoint_handlers(outputs_folder)
        if use_latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        for epoch_index in range(num_epochs):
            for fps_batch in dataset:
                actual_losses,gradients = self.train_step(fps_batch,losses_tuple)

            processed_fps_progress_to_folder(epoch_index+1,num_images,num_epochs,outputs_folder,num_progress_images,self.fps_shape,self.encode_decode_fps)
            self.data_to_tensorboard(actual_losses,gradients,epoch_index,tf_summary_writer)
            self.save_checkpoint(epoch_index+1,checkpoints_frecuency)

        log_training_end(start_time,num_epochs)

    def encode_decode_fps(self,fps_batch,is_training=False):
        mean_logvar = self.encoder(fps_batch,training=is_training)
        mean,logvar = tf.split(mean_logvar,2,1)
        z = self.sampler(mean,logvar)
        fps_processed = self.decoder(z,training=is_training)
        return mean,logvar,fps_processed

    def data_to_tensorboard(self,actual_losses,gradients,epoch_index,tf_summary_writer):
        with tf_summary_writer.as_default():
            for loss_k in actual_losses:
                tf.summary.scalar(loss_k,actual_losses[loss_k],step=epoch_index)
            tf.summary.scalar(total_loss_k,actual_losses[total_loss_k],step=epoch_index)
            #tf.summary.histogram("Gradientes",gradients,step=epoch_index)

    def save_checkpoint(self,epoch_index,checkpoints_frecuency):
        if( epoch_index%checkpoints_frecuency == 0 ):
            self.checkpoint_manager.save()

    def create_checkpoint_handlers(self,folder_name):
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                                   sampler=self.sampler,
                                                   decoder=self.decoder,
                                                   optimizer=self.optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                                  folder_name+cu.checkpoints_folder_name,
                                                                  max_to_keep=max_checkpoints_to_keep)

class GAN_CVAE():
    def __init__(self,enc_config,dec_config,disc_config,config,run_description):
        super(GAN_CVAE,self).__init__()

        self.run_description = run_description

        self.fps_shape = config[fps_shape_k]

        self.encoder = cdb.encoder_module(enc_config)
        self.sampler = cl.Sample()
        self.decoder = cdb.decoder_module(dec_config)
        self.discriminator = cdb.encoder_module(disc_config)

        self.cvae_optimizer = tf.keras.optimizers.Adam()

        self.disc_optimizer = tf.keras.optimizers.Adam()
        self.gen_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(self,fps_batch):
        actual_info,actual_gradients = {},{}

        with tf.GradientTape() as disc_batch_tape, tf.GradientTape() as disc_processed_tape, tf.GradientTape() as gen_tape:
            _,_,fps_processed = self.encode_decode_fps(fps_batch,True)

            fps_batch_logits = self.discriminator(fps_batch,training=True)
            fps_processed_logits = self.discriminator(fps_processed,training=True)

            ones_batch = tf.ones_like(fps_batch_logits)
            zeros_batch = tf.zeros_like(ones_batch)

            actual_info[gan_disc_batch_loss_k] = binary_crossentropy(ones_batch,fps_batch_logits)
            actual_info[gan_disc_processed_loss_k] = binary_crossentropy(zeros_batch,fps_processed_logits)
            actual_info[gan_gen_loss_k] = binary_crossentropy(ones_batch,fps_processed_logits)

            actual_info[gan_disc_batch_accuracy_k] = tf.keras.metrics.binary_accuracy(ones_batch,tf.keras.activations.sigmoid(fps_batch_logits))
            actual_info[gan_disc_processed_accurac_k] = tf.keras.metrics.binary_accuracy(zeros_batch,tf.keras.activations.sigmoid(fps_processed_logits))

        actual_gradients[disc_batch_gradients_k] = disc_batch_tape.gradient(actual_info[gan_disc_batch_loss_k],self.discriminator.trainable_variables)
        actual_gradients[disc_processed_gradients_k] = disc_processed_tape.gradient(actual_info[gan_disc_processed_loss_k],self.discriminator.trainable_variables)

        cvae_trainable_variables = self.encoder.trainable_variables+self.decoder.trainable_variables
        actual_gradients[gen_gradients_k] = gen_tape.gradient(actual_info[gan_gen_loss_k],cvae_trainable_variables)

        self.disc_optimizer.apply_gradients(zip(actual_gradients[disc_batch_gradients_k],self.discriminator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(actual_gradients[disc_processed_gradients_k],self.discriminator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(actual_gradients[gen_gradients_k],cvae_trainable_variables))
        return actual_info,actual_gradients

    def train(self,train_conf):
        print("GAN CVAE training started")
        start_time = time.time()

        dataset = train_conf[dataset_k]
        num_epochs = train_conf[num_epochs_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        checkpoints_frecuency = train_conf[checkpoints_frecuency_k]
        num_images = train_conf[num_images_k]
        disc_learning_rate = train_conf[disc_learning_rate_k]
        gen_learning_rate = train_conf[gen_learning_rate_k]

        self.disc_optimizer.learning_rate = disc_learning_rate
        self.gen_optimizer.learning_rate = gen_learning_rate

        outputs_folder = cu.create_output_folders("GAN_CVAE",self.run_description)
        tf_summary_writer = cu.tf_summary_writer(outputs_folder)

        self.create_checkpoint_handlers(outputs_folder)
        if use_latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        for epoch_index in range(num_epochs):
            for fps_batch in dataset:
                actual_info,actual_gradients = self.train_step(fps_batch)

            gans_data_to_tensorboard(actual_info,actual_gradients,epoch_index,tf_summary_writer)
            processed_fps_progress_to_folder(epoch_index+1,num_images,num_epochs,outputs_folder,num_progress_images,self.fps_shape,self.encode_decode_fps)
            self.save_checkpoint(epoch_index+1,checkpoints_frecuency)

        log_training_end(start_time,num_epochs)

    def encode_decode_fps(self,fps_batch,is_training=False):
        mean_logvar = self.encoder(fps_batch,training=is_training)
        mean,logvar = tf.split(mean_logvar,2,1)
        z = self.sampler(mean,logvar)
        fps_processed = self.decoder(z,training=is_training)
        return mean,logvar,fps_processed

    def save_checkpoint(self,epoch_index,checkpoints_frecuency):
        if( epoch_index%checkpoints_frecuency == 0 ):
            self.checkpoint_manager.save()

    def create_checkpoint_handlers(self,folder_name):
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                                       sampler=self.sampler,
                                                       decoder=self.decoder,
                                                       cvae_optimizer=self.cvae_optimizer,
                                                       discriminator=self.discriminator,
                                                       disc_optimizer=self.disc_optimizer,
                                                       gen_optimizer=self.gen_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                                      folder_name+cu.checkpoints_folder_name,
                                                                      max_to_keep=max_checkpoints_to_keep)

class GAN():
    def __init__(self,gen_config,disc_config,config,run_description):
        super(GAN,self).__init__()

        self.run_description = run_description

        self.fps_shape = config[fps_shape_k]
        self.batch_size = config[batch_size_k]
        self.latent_dim = config[latent_dim_k]

        self.generator = cdb.decoder_module(gen_config)
        self.discriminator = cdb.encoder_module(disc_config)

        self.disc_optimizer = tf.keras.optimizers.Adam()
        self.gen_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(self,fps_batch,):
        actual_info,actual_gradients = {},{}

        with tf.GradientTape() as disc_batch_tape, tf.GradientTape() as disc_generated_tape, tf.GradientTape() as gen_tape:
            z = tf.random.normal([self.batch_size,self.latent_dim])
            fps_generated = self.generator(z,training=True)

            fps_batch_logits = self.discriminator(fps_batch,training=True)
            fps_generated_logits = self.discriminator(fps_generated,training=True)

            ones_batch = tf.ones_like(fps_batch_logits)
            zeros_batch = tf.zeros_like(ones_batch)

            actual_info[gan_disc_batch_loss_k] = binary_crossentropy(ones_batch,fps_batch_logits)
            actual_info[gan_disc_generated_loss_k] = binary_crossentropy(zeros_batch,fps_generated_logits)
            actual_info[gan_gen_loss_k] = binary_crossentropy(ones_batch,fps_generated_logits)

            actual_info[gan_disc_batch_accuracy_k] = tf.keras.metrics.binary_accuracy(ones_batch,tf.keras.activations.sigmoid(fps_batch_logits))
            actual_info[gan_disc_generated_accurac_k] = tf.keras.metrics.binary_accuracy(zeros_batch,tf.keras.activations.sigmoid(fps_generated_logits))

        actual_gradients[disc_batch_gradients_k] = disc_batch_tape.gradient(actual_info[gan_disc_batch_loss_k],self.discriminator.trainable_variables)
        actual_gradients[disc_generated_gradients_k] = disc_generated_tape.gradient(actual_info[gan_disc_generated_loss_k],self.discriminator.trainable_variables)
        actual_gradients[gen_gradients_k] = gen_tape.gradient(actual_info[gan_gen_loss_k],self.generator.trainable_variables)

        self.disc_optimizer.apply_gradients(zip(actual_gradients[disc_batch_gradients_k],self.discriminator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(actual_gradients[disc_generated_gradients_k],self.discriminator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(actual_gradients[gen_gradients_k],self.generator.trainable_variables))
        return actual_info,actual_gradients

    def train(self,train_conf):
        print("GAN training started")
        start_time = time.time()

        dataset = train_conf[dataset_k]
        num_epochs = train_conf[num_epochs_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        checkpoints_frecuency = train_conf[checkpoints_frecuency_k]
        num_images = train_conf[num_images_k]
        disc_learning_rate = train_conf[disc_learning_rate_k]
        gen_learning_rate = train_conf[gen_learning_rate_k]

        self.disc_optimizer.learning_rate = disc_learning_rate
        self.gen_optimizer.learning_rate = gen_learning_rate

        outputs_folder = cu.create_output_folders("GAN",self.run_description)
        tf_summary_writer = cu.tf_summary_writer(outputs_folder)

        self.create_checkpoint_handlers(outputs_folder)
        if use_latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        for epoch_index in range(num_epochs):
            for fps_batch in dataset:
                actual_info,actual_gradients = self.train_step(fps_batch)

            gans_data_to_tensorboard(actual_info,actual_gradients,epoch_index,tf_summary_writer)
            self.generated_fps_progress_to_folder(epoch_index+1,num_images,num_epochs,outputs_folder,num_progress_images)
            self.save_checkpoint(epoch_index+1,checkpoints_frecuency)

        log_training_end(start_time,num_epochs)

    def save_checkpoint(self,epoch_index,checkpoints_frecuency):
        if( epoch_index%checkpoints_frecuency == 0 ):
            self.checkpoint_manager.save()

    def create_checkpoint_handlers(self,folder_name):
        self.checkpoint = tf.train.Checkpoint(decoder=self.decoder,
                                                  gen_optimizer=self.gen_optimizer,
                                                  discriminator=self.discriminator,
                                                  disc_optimizer=self.disc_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                                 folder_name+cu.checkpoints_folder_name,
                                                                 max_to_keep=max_checkpoints_to_keep)

    def generated_fps_progress_to_folder(self,epoch_index,num_images,num_epochs,outputs_folder,num_fps):

        if( epoch_index%(num_epochs/num_images) == 0 ):
            z = dp.load_verification_noises(num_fps,self.latent_dim).numpy()
            fps_generated = self.generator(z,training=False).numpy()

            fps_generated_shapes = np.shape(fps_generated)
            if fps_processed_shapes[3] == 1:
                fps_generated = np.reshape(fps_generated,(fps_generated_shapes[0],fps_generated_shapes[1],fps_generated_shapes[2]))

            fig,axs = plt.subplots(num_fps,1,figsize=(5,5),constrained_layout=True)
            fig.suptitle("Epoch: {}".format(epoch_index))

            for i in range(num_fps):
                axs[i,0].imshow(fps_generated[i,:],cmap="gray")
                axs[i,0].axis('off')

            plt.savefig(outputs_folder+cu.performance_imgs_folder_name+"/fp_at_epoch_{}.png".format(epoch_index),bbox_inches='tight')
            plt.close(fig)

# GENERAL METHODS
def processed_fps_progress_to_folder(epoch_index,num_images,num_epochs,outputs_folder,num_fps,fps_shape,encode_decode_fps):

    if( epoch_index%(num_epochs/num_images) == 0 ):
        fps_batch = dp.load_verification_images(fps_shape,num_fps).numpy()
        _,_,fps_processed = encode_decode_fps(fps_batch)
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

        plt.savefig(outputs_folder+cu.performance_imgs_folder_name+"/fp_at_epoch_{}.png".format(epoch_index),bbox_inches='tight')
        plt.close(fig)

def gans_data_to_tensorboard(actual_info,actual_gradients,epoch_index,tf_summary_writer):
    with tf_summary_writer.as_default():
        for key in actual_info:
            tf.summary.scalar(key,actual_info[key],step=epoch_index)
        for key in actual_gradients:
            pass#tf.summary.histogram(key,actual_gradients,step=epoch_index)

def log_training_end(start_time,num_epochs):
    print("Training total time: "+str(np.round(time.time()-start_time,2)))
    print("Average time per epoch: "+str(np.round((time.time()-start_time)/num_epochs,2)))
    print("Training finished")