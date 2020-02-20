import conv_deconv_blocks as cdb
import matplotlib.pyplot as plt
import custom_layers as cl
import tensorflow as tf
import numpy as np
import time

# Dictionary keys
adam_alpha_k = "adam_alpha" #1e-4
use_total_variation_k = "use_total_variation" #True
lambda_total_variation_k = "lambda_total_variation" #0.5
checkpoints_folder_k = "checkpoints_folder"
max_checkpoints_k = "max_checkpoints" #2
dataset_k = "dataset"
use_latest_checkpoint_k = "use_latest_checkpoint"
num_epochs_k = "num_epochs"
percent_progress_savings_k = "percent_progress_savings"
time_measures_k = "time_measures"
num_images_k = "num_images"
num_rows_verification_k = "num_rows_verification"
images_folder_k = "images_folder"
batch_size_k = "batch_size"
noise_shape_k = "noise_shape"
create_checkpoints_k = "create_checkpoints"

# Losses
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class DCGAN():
    def __init__(self,gen_config,disc_config,config):

        self.generator = cdb.decoder_module(None,gen_config)
        self.discriminator = cdb.encoder_module(None,disc_config)

        self.generator_optimizer = tf.keras.optimizers.Adam(config[adam_alpha_k])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(config[adam_alpha_k])

        self.use_total_variation = config[use_total_variation_k]
        self.lambda_total_variation = config[lambda_total_variation_k]

        self.tf_checkpoint = tf.train.Checkpoint(generator=self.generator,
                                                 discriminator=self.discriminator,
                                                 generator_optimizer=self.generator_optimizer,
                                                 discriminator_optimizer=self.discriminator_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(self.tf_checkpoint,config[checkpoints_folder_k],max_to_keep=config[max_checkpoints_k])

        self.num_rows_verification = config[num_rows_verification_k]
        self.num_verification_images = np.power(self.num_rows_verification,2)
        self.images_folder = config[images_folder_k]
        self.batch_size = config[batch_size_k]
        self.noise_shape = config[noise_shape_k]
        self.verification_noises = tf.random.normal([self.num_verification_images,self.noise_shape])
        self.costs = {"g":[],"d":[]}

    def generator_loss(self,fake_output):
        return binary_cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self,real_output, fake_output):
        real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self,images_batch):
        noise = tf.random.normal([self.batch_size,self.noise_shape])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise,training=True)

            real_disc_output = self.discriminator(images_batch,training=True)
            fake_disc_output = self.discriminator(generated_images,training=True)

            gen_loss = self.generator_loss(fake_disc_output)
            disc_loss = self.discriminator_loss(real_disc_output,fake_disc_output)

            if self.use_total_variation:
                disc_loss += self.lambda_total_variation*tf.reduce_mean(tf.image.total_variation(generated_images))

        generator_gradients = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self,train_conf):

        dataset = train_conf[dataset_k]
        use_latest_checkpoint = train_conf[use_latest_checkpoint_k]
        num_epochs = train_conf[num_epochs_k]
        percent_progress_savings = train_conf[percent_progress_savings_k]
        time_measures = train_conf[time_measures_k]
        num_images = train_conf[num_images_k]
        create_checkpoints = train_conf[create_checkpoints_k]

        if use_latest_checkpoint:
            self.tf_checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        epochs_progess_savings = [int((percent/100)*num_epochs) for percent in percent_progress_savings]

        for epoch_index in range(num_epochs):

            start = time.time()

            for images_batch in dataset:

                gen_loss, disc_loss = self.train_step(images_batch)
                self.costs["g"].append(gen_loss)
                self.costs["d"].append(disc_loss)

            epoch_index_1 = epoch_index + 1

            if epoch_index_1%time_measures == 0:
                print ('Time for epoch {} is {} sec'.format(epoch_index_1, np.round(time.time()-start,2)))

            if epoch_index_1%int(num_epochs/num_images) == 0:
                self.save_verification_images(epoch_index,use_latest_checkpoint)

            if create_checkpoints:
                if epoch_index_1 in epochs_progess_savings:
                    self.checkpoint_manager.save()

    def graph_costs(self):
        plt.subplot(1,2,1)
        plt.plot(self.costs["g"])
        plt.title("Generator Loss")

        plt.subplot(1,3,2)
        plt.plot(self.costs["d"])
        plt.title("Discriinator Loss")

    def save_verification_images(self,epoch,use_latest_checkpoint):
        generated_images = self.generator(self.verification_noises).numpy()

        for im_index in range(self.num_verification_images):
            plt.subplot(self.num_rows_verification,self.num_rows_verification,im_index+1)
            act_image = generated_images[im_index,:]

            cmap = None
            if(act_image.shape[2]==1):
                cmap = "gray"
                act_image = act_image.reshape(act_image.shape[0],act_image.shape[1])

            plt.imshow(act_image,cmap=cmap)
            plt.axis('off')

        image_dir = self.images_folder + "image_at_epoch_{}.png".format(epoch+1)

        if use_latest_checkpoint:
            image_dir = self.images_folder + "image_at_epoch_{}_model_restored.png".format(epoch+1)

        plt.savefig(image_dir)

    def generate_fp(self):
        noise = np.random.randn(1,self.noise_shape)
        generated_image = self.generator(noise).numpy()
        return generated_image
