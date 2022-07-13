

from __future__ import print_function, division

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import RMSprop
from keras.optimizers import rmsprop_v2

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import tensorflow as tf
# from tensorflow.python.keras.optimizers import rmsprop_v2
from functools import partial
import pandas as pd
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = rmsprop_v2.RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #
        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        data=pd.read_csv('adjProteinDrug.csv',index_col=None,header=None)
        X_train = load_data(data,len(data))
        print(X_train.shape)
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],[valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # def sample_image(self, epoch):
    #     # r, c = 8, 8
    #     noise = np.random.normal(-1, 1, (1, self.latent_dim))
    #     gen_imgs = self.generator.predict(noise)
    #     gen_imgs = np.squeeze(gen_imgs, axis=3)  # (5464,8,8)/(10,554,554)
    #     # print(gen_imgs.shape)
    #     c = []
    #     for i in range(len(gen_imgs)):
    #         # c.append(gen_imgs[i].reshape(1, 64))  # 对嵌入特征
    #         c.append(gen_imgs[i]) #对邻接矩阵
    #     final = np.vstack(c)
    #     # print(final.shape)
    #     pd.DataFrame(final).to_csv('./adj.csv', index=0, header=0)
    def sample_images(self, epoch):
        # r, c = 8, 8
        noise = np.random.normal(0, 1, (554, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = np.squeeze(gen_imgs, axis=3)  # (544,8,8)/(10,554,554)
        # print(gen_imgs.shape)
        c = []
        for i in range(len(gen_imgs)):
            real_adj = gen_imgs[i][6:21, 6:21]  # 从噪声序列28x28的子序列中获得中心15x15的矩阵(左闭右开)
            # 将28x28还原成8x8的矩阵，并重塑为1x64
            c.append(real_adj.reshape(1, 225))  # 对嵌入特征
            # c.append(gen_imgs[i]) #对邻接矩阵
        final = np.vstack(c)
        # print(final.shape)
        pd.DataFrame(final).to_csv('./simulate_adj.csv', index=0, header=0)

# def load_dataAdj(data,length): #转为小矩阵
#     a=[]
#     for i in range(0,64): #存储554个x的矩阵
#         c = []
#         for j in range(0,length):
#             locs1=data.iloc[j].values
#             locs=np.pad(locs1,(0,length-len(locs1)))
#             c.append(locs)
#             # print(len(locs))
#         a.append(c)
#     b=[]
#     for j in a:
#         b.append(j)
#     df=np.array(b)
#     return np.expand_dims(df,axis=3)
def load_data(data,length): #转为小矩阵，读取数据为554x225
    a=[]
    for i in range(0,length):
        c=[]
        for j in range(0,225,15):
          c.append(data.iloc[i,j:j+15].values)
        c1=np.array(c) #转为数组
        x=np.pad(c1, ((6, 7), (6, 7)), 'constant', constant_values=(0, 0)) #将8x8数组转换为28x28的数组
        c=list(x)
        a.append(c)
    b=[]
    for j in a:
        b.append(j)
        # pd.DataFrame(j).to_csv('./替代数据.csv')
    df=np.array(b)
    return np.expand_dims(df,axis=3)


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=2000, batch_size=32, sample_interval=10)
