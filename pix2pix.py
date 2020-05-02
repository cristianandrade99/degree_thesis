import matplotlib.pyplot as plt
import data_processing as dp
import cris_utils as cu
import tensorflow as tf
import numpy as np
import keys as km
import time

class Pix2Pix():
    def __init__(self,general_config,gen_disc_config,generator_discriminator):

        self.general_config = general_config
        self.gen_disc_config = gen_disc_config

        self.generator,self.discriminator = generator_discriminator

        self.gen_optimizer = tf.keras.optimizers.Adam()
        self.disc_optimizer = tf.keras.optimizers.Adam()

        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self,fps_to_enhance,fps_target,train_data):
        actual_info,actual_gradients,actual_accuracies = {},{},{}

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_target_tape, tf.GradientTape() as disc_enhanced_tape:
            fps_enhanced = self.generator(fps_to_enhance, training=True)

            fps_target_logits = self.discriminator([fps_to_enhance,fps_target],training=True)
            fps_enhanced_logits = self.discriminator([fps_to_enhance,fps_enhanced],training=True)

            ones_loss,alphas_ones_loss,zeros_loss = train_data[km.entropy_p_loss_k]
            ones_acc,alphas_ones_acc,zeros_acc = train_data[km.entropy_p_acc_k]
            losses_tuple = train_data[km.losses_tuple_k]
            gen_loss_alph,disc_loss_alph = train_data[km.gen_disc_loss_alphas_k][0],train_data[km.gen_disc_loss_alphas_k][1]

            actual_losses = self.calc_losses(losses_tuple,fps_enhanced,fps_target)
            actual_losses[km.gen_loss_k] = self.binary_crossentropy(ones_loss,fps_enhanced_logits)
            total_gen_loss = gen_loss_alph*(actual_losses[km.total_loss_k] + actual_losses[km.gen_loss_k])

            actual_losses[km.disc_target_loss_k] = disc_loss_alph*self.binary_crossentropy(alphas_ones_loss,fps_target_logits)
            actual_losses[km.disc_enhanced_loss_k] = disc_loss_alph*self.binary_crossentropy(zeros_loss,fps_enhanced_logits)

            target_probs = tf.reduce_mean( tf.keras.activations.sigmoid(fps_target_logits),[1,2,3] )
            enhanced_probs = tf.reduce_mean( tf.keras.activations.sigmoid(fps_enhanced_logits),[1,2,3] )

            actual_accuracies[km.disc_target_accuracy_k] = tf.keras.metrics.binary_accuracy(ones_acc,target_probs)
            actual_accuracies[km.disc_enhanced_accurac_k] = tf.keras.metrics.binary_accuracy(zeros_acc,enhanced_probs)

            actual_gradients[km.gen_gradients_k] = gen_tape.gradient(total_gen_loss,self.generator.trainable_variables)
            actual_gradients[km.disc_target_gradients_k] = disc_target_tape.gradient(actual_losses[km.disc_target_loss_k],self.discriminator.trainable_variables)
            actual_gradients[km.disc_enhanced_gradients_k] = disc_enhanced_tape.gradient(actual_losses[km.disc_enhanced_loss_k],self.discriminator.trainable_variables)

            self.gen_optimizer.apply_gradients(zip(actual_gradients[km.gen_gradients_k],self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(actual_gradients[km.disc_target_gradients_k],self.discriminator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(actual_gradients[km.disc_enhanced_gradients_k],self.discriminator.trainable_variables))

            return actual_losses,actual_gradients,actual_accuracies

    def train(self,train_conf):

        num_epochs = train_conf[km.num_epochs_k]
        losses_tuple = train_conf[km.types_losses_k],train_conf[km.alphas_losses_k]
        use_latest_checkpoint = train_conf[km.use_latest_checkpoint_k]
        epochs_to_save = train_conf[km.epochs_to_save_k]
        num_images = train_conf[km.num_images_k]
        gen_adam_params = train_conf[km.gen_adam_params_k]
        disc_adam_params = train_conf[km.disc_adam_params_k]
        gen_disc_loss_alphas = train_conf[km.gen_disc_loss_alphas_k]
        alpha_ones_p = train_conf[km.alpha_ones_p_k]
        num_histograms = train_conf[km.num_histograms_k]
        dataset = train_conf[km.data_info_k]
        self.num_progress_images = train_conf[km.num_progress_images_k]

        outputs_folder = cu.outputs_folder
        tf_summary_writer = cu.tf_summary_writer(outputs_folder)

        cu.cu_print("Pix2Pix model training started")
        start_time = time.time()

        msg_config = cu.createDictMsg(self.general_config,"=== General Configuration ===")
        msg_config += cu.createDictMsg(self.gen_disc_config,"=== Generator Discriminator Configuration ===")
        msg_config += cu.createDictMsg(train_conf,"=== Training Configuration ===")
        cu.cu_print(msg_config)

        self.gen_optimizer.learning_rate = gen_adam_params[0]
        self.disc_optimizer.learning_rate = disc_adam_params[0]

        self.gen_optimizer.beta_1 = gen_adam_params[1]
        self.disc_optimizer.beta_1 = disc_adam_params[1]

        self.config_to_tensorboard(tf_summary_writer,msg_config)

        self.create_checkpoint_handlers(outputs_folder)
        if use_latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        train_data = {}
        train_data[km.entropy_p_loss_k] = self.entropy_p_vectors([self.general_config[km.batch_size_k],29,29,1],alpha_ones_p)
        train_data[km.entropy_p_acc_k] = self.entropy_p_vectors([self.general_config[km.batch_size_k],],alpha_ones_p)
        train_data[km.losses_tuple_k] = losses_tuple
        train_data[km.gen_disc_loss_alphas_k] = gen_disc_loss_alphas

        for epoch_index in range(num_epochs):
            for fps_to_enhance,fps_target in dataset:
                train_step_info = self.train_step(fps_to_enhance,fps_target,train_data)

            self.p2p_data_to_tensorboard(train_step_info,epoch_index,num_epochs,num_histograms,tf_summary_writer)
            self.enhanced_fps_progress_to_folder(self.general_config[km.fps_shape_k],num_images,outputs_folder,epoch_index,num_epochs)
            self.save_checkpoint(epoch_index,epochs_to_save)

        self.log_training_end(start_time,num_epochs)

    def create_checkpoint_handlers(self,folder_name):
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              gen_optimizer=self.gen_optimizer,
                                              disc_optimizer=self.disc_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory="./{}/{}".format(folder_name,cu.checkpoints_folder_name),
                                                             max_to_keep = None,
                                                             checkpoint_name='epoch')

    def p2p_data_to_tensorboard(self,train_step_info,epoch_index,num_epochs,num_histograms,tf_summary_writer):
        actual_losses,actual_gradients,actual_accuracies = train_step_info

        with tf_summary_writer.as_default():

            for key in actual_losses:
                tf.summary.scalar(key,tf.squeeze(actual_losses[key]),step=epoch_index)

            for key in actual_accuracies:
                tf.summary.scalar(key,tf.squeeze(actual_accuracies[key]),step=epoch_index)

            n_hist = num_histograms if num_epochs>=num_histograms else num_epochs
            if( n_hist != 0 and epoch_index%int(num_epochs/n_hist) == 0 ):

                for tv,g in zip(self.generator.trainable_variables,actual_gradients[km.gen_gradients_k]):
                    tf.summary.histogram(tv.name+" gradient",g,step=epoch_index)

                for tv,g in zip(self.discriminator.trainable_variables,actual_gradients[km.disc_target_gradients_k]):
                    tf.summary.histogram(tv.name+" target gradient",g,step=epoch_index)

                for tv,g in zip(self.discriminator.trainable_variables,actual_gradients[km.disc_enhanced_gradients_k]):
                    tf.summary.histogram(tv.name+" enhanced gradient",g,step=epoch_index)

    def enhanced_fps_progress_to_folder(self,fps_shape,num_images,outputs_folder,epoch_index,num_epochs):
        n_images = num_images if num_epochs>=num_images else num_epochs

        if( n_images != 0 and epoch_index%int(num_epochs/n_images) == 0 ):
            fps_to_enhance,fps_target = dp.load_verification_images(fps_shape,self.num_progress_images)
            fps_enhanced = self.generator(fps_to_enhance,training=False).numpy()
            self.save_enhanced_fps(fps_to_enhance,fps_enhanced,fps_target,outputs_folder,epoch_index)

    def save_checkpoint(self,epoch_index,epochs_to_save):
        if( epoch_index in epochs_to_save ):
            self.checkpoint_manager.save(epoch_index)

    def calc_losses(self,losses_tuple,batch_1,batch_2,dicc_info=None):
        losses,alphas = losses_tuple
        actual_losses = {}
        total_loss = 0.0
        for loss,alph in zip(losses,alphas):
            if loss == km.square_loss:
                actual_losses[km.square_loss] = alph*tf.reduce_mean(tf.keras.losses.MSE(batch_1,batch_2))
            elif loss == km.kl_loss:
                actual_losses[km.kl_loss] = alph*tf.reduce_mean(0.5*( tf.square(dicc_info[km.mean_k])+tf.exp(dicc_info[km.logvar_k])-1-logvar ))
            elif loss == km.ssim_loss:
                actual_losses[km.ssim_loss] = alph*tf.reduce_mean( tf.image.ssim(batch_1,batch_2,max_val=2.0) )
            elif loss == km.tv_loss:
                actual_losses[km.tv_loss] = alph*tf.reduce_mean( tf.image.total_variation(batch_2) )
            elif loss == km.cross_loss:
                actual_losses[km.cross_loss] = alph*self.binary_crossentropy((batch_1+1)/2,(batch_2+1)/2)
            elif loss == km.l1_loss:
                actual_losses[km.l1_loss] = alph*tf.reduce_mean( tf.abs(batch_2-batch_1) )

            total_loss += actual_losses[loss]

        actual_losses[km.total_loss_k] = total_loss

        return actual_losses

    def save_enhanced_fps(self,fps_to_enhance,fps_enhanced,fps_target,outputs_folder,epoch_index):

        if np.shape(fps_to_enhance)[3] == 1:
            fps_to_enhance = np.squeeze(fps_to_enhance,axis=3)
            fps_enhanced = np.squeeze(fps_enhanced,axis=3)
            fps_target = np.squeeze(fps_target,axis=3)

        fig,axs = plt.subplots(self.num_progress_images,1,figsize=(30,30),constrained_layout=True)
        fig.suptitle("Epoch: {}".format(epoch_index))

        min,max = np.min(fps_enhanced),np.max(fps_enhanced)
        fps_enhanced_m = -1 + 2*(fps_enhanced-min)/(max-min)
        fps = np.concatenate((fps_to_enhance,fps_enhanced_m,fps_target),2)
        for i in range(self.num_progress_images):
            axs[i].imshow(fps[i,:],cmap="gray")
            axs[i].axis('off')

        plt.savefig("./{}/{}/fp_at_epoch_{}".format(outputs_folder,cu.performance_imgs_folder_name,epoch_index),bbox_inches='tight')
        plt.close(fig)

    def config_to_tensorboard(self,tf_summary_writer,config):
        with tf_summary_writer.as_default():
            tf.summary.text("Configuration",config,step=0)

    def log_training_end(self,start_time,num_epochs):
        cu.cu_print("Training total time: "+str(np.round(time.time()-start_time,2)))
        cu.cu_print("Average time per epoch: "+str(np.round((time.time()-start_time)/num_epochs,2)))
        cu.cu_print("Training finished")
        cu.close_log()

    def entropy_p_vectors(self,size,alpha_ones_p):
        ones = tf.ones(size)
        alphas_ones = alpha_ones_p*tf.ones_like(ones)
        zeros = tf.zeros_like(ones)
        return ones,alphas_ones,zeros
