import conv_deconv_blocks as cdb
import matplotlib.pyplot as plt
import data_processing as dp
import custom_layers as cl
import cris_utils as cu
import tensorflow as tf
import datetime as dt
import numpy as np
import time

fps_shape_k = cdb.fps_shape_k
latent_dim_k = "latent_dim"
batch_size_k = "batch_size"
data_dir_patt_k = "data_dir_patt"

use_latest_checkpoint_k = "use_latest_checkpoint"
num_epochs_k = "num_epochs_k"
checkpoints_frecuency_k = "checkpoints_frecuency"
num_images_k = "num_images"
types_losses_k = "types_losses"
alphas_losses_k = "alphas_losses_k"
num_histograms_k = "num_histograms"
data_info_k = "data_info"

disc_adam_params_k = "disc_adam_params"
gen_adam_params_k = "gen_adam_params"

learning_rate_k = "learning_rate"
disc_learning_rate_k = "disc_learning_rate"
gen_learning_rate_k = "gen_learning_rate"
alpha_ones_p_k = "alpha_ones"
entropy_p_loss_k = "entropy_p_loss"
entropy_p_acc_k = "entropy_p_acc"
losses_tuple_k = "losses_tuple"

mean_k = "mean_k"
logvar_k = "logvar_k"

cvae_model = "cvae_model"
gan_model = "gan_model"

square_loss = "Mean Squared Error"
l1_loss = "L1 Loss"
kl_loss = "KL Divergence"
ssim_loss = "SSIM Loss"
tv_loss = "Total Variation Loss"
cross_loss = "CrossEntropy Loss"
total_loss_k = "Total Loss"

disc_target_loss_k = "Discriminator Target Loss"
disc_enhanced_loss_k = "Discriminator Enhanced Loss"
gen_loss_k = "Generator Loss"

disc_target_accuracy_k = "Discriminator Target Accuracy"
disc_enhanced_accurac_k = "Discriminator Enhanced Accuracy"

disc_target_gradients_k = "Discriminator Target Gradients"
disc_enhanced_gradients_k = "Discriminator Enhanced Gradients"
gen_gradients_k = "Generator Gradients"

binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
max_checkpoints_to_keep = 2
num_progress_images = 4

class P2P():
    def __init__(self,generator,discriminator,config,run_description):

        self.generator = generator
        self.discriminator = discriminator

        self.config = config
        self.run_description = run_description

        self.gen_optimizer = tf.keras.optimizers.Adam()
        self.disc_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(self,fps_to_enhance,fps_target,train_data):
        actual_info,actual_gradients,actual_accuracies = {},{},{}

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_target_tape, tf.GradientTape() as disc_enhanced_tape:
            fps_enhanced = self.generator(fps_to_enhance, training=True)

            fps_target_logits = self.discriminator([fps_to_enhance,fps_target],training=True)
            fps_enhanced_logits = self.discriminator([fps_to_enhance,fps_enhanced],training=True)

            ones_loss,alphas_ones_loss,zeros_loss = train_data[entropy_p_loss_k]
            ones_acc,alphas_ones_acc,zeros_acc = train_data[entropy_p_acc_k]
            losses_tuple = train_data[losses_tuple_k]

            actual_losses = calc_losses(losses_tuple,fps_to_enhance,fps_enhanced)
            actual_losses[gen_loss_k] = binary_crossentropy(ones_loss,fps_enhanced_logits)
            total_gen_loss = actual_losses[total_loss_k] + actual_losses[gen_loss_k]

            actual_losses[disc_target_loss_k] = binary_crossentropy(alphas_ones_loss,fps_target_logits)
            actual_losses[disc_enhanced_loss_k] = binary_crossentropy(zeros_loss,fps_enhanced_logits)

            target_probs = tf.reduce_mean( tf.keras.activations.sigmoid(fps_target_logits),[1,2,3] )
            enhanced_probs = tf.reduce_mean( tf.keras.activations.sigmoid(fps_enhanced_logits),[1,2,3] )

            actual_accuracies[disc_target_accuracy_k] = tf.keras.metrics.binary_accuracy(ones_acc,target_probs)
            actual_accuracies[disc_enhanced_accurac_k] = tf.keras.metrics.binary_accuracy(zeros_acc,enhanced_probs)

            actual_gradients[gen_gradients_k] = gen_tape.gradient(total_gen_loss,self.generator.trainable_variables)
            actual_gradients[disc_target_gradients_k] = disc_target_tape.gradient(actual_losses[disc_target_loss_k],self.discriminator.trainable_variables)
            actual_gradients[disc_enhanced_gradients_k] = disc_enhanced_tape.gradient(actual_losses[disc_enhanced_loss_k],self.discriminator.trainable_variables)

            self.gen_optimizer.apply_gradients(zip(actual_gradients[gen_gradients_k],self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(actual_gradients[disc_target_gradients_k],self.discriminator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(actual_gradients[disc_enhanced_gradients_k],self.discriminator.trainable_variables))

            return actual_losses,actual_gradients,actual_accuracies

    def train(self,train_conf):

        num_epochs = train_conf[num_epochs_k]
        losses_tuple = train_conf[types_losses_k],train_conf[alphas_losses_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        checkpoints_frecuency = train_conf[checkpoints_frecuency_k]
        num_images = train_conf[num_images_k]
        gen_adam_params = train_conf[gen_adam_params_k]
        disc_adam_params = train_conf[disc_adam_params_k]
        alpha_ones_p = train_conf[alpha_ones_p_k]
        num_histograms = train_conf[num_histograms_k]
        dataset = train_conf[data_info_k]

        outputs_folder = cu.outputs_folder
        self.tf_summary_writer = cu.tf_summary_writer(outputs_folder)

        cu.cu_print("P2P training started")
        start_time = time.time()

        msg_config = cu.printDict(self.config,"Model Configuration")
        msg_config += cu.printDict(train_conf,"Train Configuration")
        cu.cu_print(msg_config)

        self.gen_optimizer.learning_rate = gen_adam_params[0]
        self.disc_optimizer.learning_rate = disc_adam_params[0]

        self.gen_optimizer.beta_1 = gen_adam_params[1]
        self.disc_optimizer.beta_1 = disc_adam_params[1]

        config_to_tensorboard(self.tf_summary_writer,msg_config)

        self.create_checkpoint_handlers(outputs_folder)
        if use_latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        train_data = {}
        train_data[entropy_p_loss_k] = entropy_p_vectors([self.config[batch_size_k],30,30,1],alpha_ones_p)
        train_data[entropy_p_acc_k] = entropy_p_vectors([self.config[batch_size_k],],alpha_ones_p)
        train_data[losses_tuple_k] = losses_tuple

        for epoch_index in range(num_epochs):
            for fps_to_enhance,fps_target in dataset:
                train_step_info = self.train_step(fps_to_enhance,fps_target,train_data)

            self.p2p_data_to_tensorboard(train_step_info,epoch_index,num_epochs,num_histograms)
            self.enhanced_fps_progress_to_folder(self.config[fps_shape_k],num_images,outputs_folder,epoch_index,num_epochs)
            self.save_checkpoint(epoch_index,num_epochs,checkpoints_frecuency)

        log_training_end(start_time,num_epochs)

    def create_checkpoint_handlers(self,folder_name):
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              gen_optimizer=self.gen_optimizer,
                                              disc_optimizer=self.disc_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             "./{}/{}".format(folder_name,cu.checkpoints_folder_name),
                                                             max_to_keep=max_checkpoints_to_keep)

    def p2p_data_to_tensorboard(self,train_step_info,epoch_index,num_epochs,num_histograms):
        actual_losses,actual_gradients,actual_accuracies = train_step_info

        with self.tf_summary_writer.as_default():

            for key in actual_losses:
                tf.summary.scalar(key,tf.squeeze(actual_losses[key]),step=epoch_index)

            for key in actual_accuracies:
                tf.summary.scalar(key,tf.squeeze(actual_accuracies[key]),step=epoch_index)

            n_hist = num_histograms if num_epochs>=num_histograms else num_epochs
            if( n_hist != 0 and epoch_index%int(num_epochs/n_hist) == 0 ):

                for tv,g in zip(self.generator.trainable_variables,actual_gradients[gen_gradients_k]):
                    tf.summary.histogram(tv.name+" gradient",g,step=epoch_index)

                for tv,g in zip(self.discriminator.trainable_variables,actual_gradients[disc_target_gradients_k]):
                    tf.summary.histogram(tv.name+" target gradient",g,step=epoch_index)

                for tv,g in zip(self.discriminator.trainable_variables,actual_gradients[disc_enhanced_gradients_k]):
                    tf.summary.histogram(tv.name+" enhanced gradient",g,step=epoch_index)

    def enhanced_fps_progress_to_folder(self,fps_shape,num_images,outputs_folder,epoch_index,num_epochs):
        n_images = num_images if num_epochs>=num_images else num_epochs

        if( n_images != 0 and epoch_index%int(num_epochs/n_images) == 0 ):
            fps_to_enhance,fps_target = dp.load_verification_images(fps_shape,num_progress_images)
            fps_enhanced = self.generator(fps_to_enhance,training=False).numpy()
            save_enhanced_fps(fps_to_enhance,fps_enhanced,fps_target,outputs_folder,epoch_index)

    def save_checkpoint(self,epoch_index,num_epochs,checkpoints_frecuency):
        check_frec = checkpoints_frecuency if num_epochs >= checkpoints_frecuency else 2
        if( check_frec != 0 and (epoch_index+1)%check_frec == 0 ):
            self.checkpoint_manager.save()

# GLOBAL METHODS
def calc_losses(losses_tuple,batch_1,batch_2,dicc_info=None):
    losses,alphas = losses_tuple
    actual_losses = {}
    total_loss = 0.0
    for loss,alph in zip(losses,alphas):
        if loss == square_loss:
            actual_losses[square_loss] = alph*tf.reduce_mean(tf.keras.losses.MSE(batch_1,batch_2))
        elif loss == kl_loss:
            actual_losses[kl_loss] = alph*tf.reduce_mean(0.5*( tf.square(dicc_info[mean_k])+tf.exp(dicc_info[logvar_k])-1-logvar ))
        elif loss == ssim_loss:
            actual_losses[ssim_loss] = alph*tf.reduce_mean( tf.image.ssim(batch_1,batch_2,max_val=2.0) )
        elif loss == tv_loss:
            actual_losses[tv_loss] = alph*tf.reduce_mean( tf.image.total_variation(batch_2) )
        elif loss == cross_loss:
            actual_losses[cross_loss] = alph*binary_crossentropy((batch_1+1)/2,(batch_2+1)/2)
        elif loss == l1_loss:
            actual_losses[l1_loss] = alph*tf.reduce_mean( tf.abs(batch_2-batch_1) )

        total_loss += actual_losses[loss]

    actual_losses[total_loss_k] = total_loss

    return actual_losses

def save_enhanced_fps(fps_to_enhance,fps_enhanced,fps_target,outputs_folder,epoch_index):

    if np.shape(fps_to_enhance)[3] == 1:
        fps_to_enhance = np.squeeze(fps_to_enhance,axis=3)
        fps_enhanced = np.squeeze(fps_enhanced,axis=3)
        fps_target = np.squeeze(fps_target,axis=3)

    fig,axs = plt.subplots(num_progress_images,1,figsize=(20,20),constrained_layout=True)
    fig.suptitle("Epoch: {}".format(epoch_index))

    min,max = np.min(fps_enhanced),np.max(fps_enhanced)
    fps_enhanced_m = -1 + 2*(fps_enhanced-min)/(max-min)
    fps = np.concatenate((fps_to_enhance,fps_enhanced_m,fps_target),2)
    for i in range(num_progress_images):
        axs[i].imshow(fps[i,:],cmap="gray")
        axs[i].axis('off')

    plt.savefig("./{}/{}/fp_at_epoch_{}".format(outputs_folder,cu.performance_imgs_folder_name,epoch_index),bbox_inches='tight')
    plt.close(fig)

def config_to_tensorboard(tf_summary_writer,config):
    with tf_summary_writer.as_default():
        tf.summary.text("Configuration",config,step=0)

def log_training_end(start_time,num_epochs):
    cu.cu_print("Training total time: "+str(np.round(time.time()-start_time,2)))
    cu.cu_print("Average time per epoch: "+str(np.round((time.time()-start_time)/num_epochs,2)))
    cu.cu_print("Training finished")
    cu.close_log()

def entropy_p_vectors(size,alpha_ones_p):
    ones = tf.ones(size)
    alphas_ones = alpha_ones_p*tf.ones_like(ones)
    zeros = tf.zeros_like(ones)
    return ones,alphas_ones,zeros
