import datetime

import numpy as np
from tensorflow.keras.layers import Input, UpSampling1D, ReLU, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, \
    AvgPool2D, \
    UpSampling2D, Concatenate, Dropout
# import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, mae

from data_loader import DataLoader


class RCAN:

    def __init__(self):
        self.kernel_size = 3
        self.filters = 32
        # self.img_shape = (472, 472, 3) # from paper
        self.img_size = 512  # power of 2 so it acc works :/
        self.output_size = 58
        self.input_shape = (self.img_size, self.img_size, 3)
        self.rgb_shape = (self.output_size, self.output_size, 3)
        self.seg_shape = (self.output_size, self.output_size, 5)
        self.depth_shape = (self.output_size, self.output_size, 1)
        self.batch_size = None
        # self.patch_sizes = [472, 236, 118]
        self.patch_sizes = [512, 256, 128]
        self.patch_weights = [1, 1, 1]
        self.epochs = None
        self.lr = None
        self.pool_size = 2

        self.sample_interval = 10
        self.model_save_interval = 100

        self.opt = Adam(lr=0.001)  # OPTIMISER??
        self.data_loader = DataLoader(dataset_name="image")

        self.g = self.define_generator()
        self.d = self.define_discriminator()
        self.gan = self.define_gan()

    def conv2d_layer(self, layer_inp, filters, strides=1, avg_pool=True, kernel_size=3):
        c = layer_inp

        # padding or nah??
        c = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(c)
        if avg_pool:
            c = AvgPool2D(pool_size=(2, 2))(c)  # CHECK ME!!
        c = BatchNormalization()(c)  # Change to instance norm
        c = ReLU()(c)

        return c

    def deconv2d_layer(self, layer_input, filters1, filters2=None, skip_input=None, use_skip_input=True,
                       upsample_size=2, strides=1,
                       strides2=1):
        print("deconv2d layer shape: ", layer_input.shape, skip_input.shape)
        # d = layer_input
        if use_skip_input:
            layer_input = Concatenate()([layer_input, skip_input])

        d = Conv2D(filters1, kernel_size=1, strides=strides, padding='same', activation='relu')(
            layer_input)
        d = BatchNormalization()(d)

        d = UpSampling2D(size=upsample_size, interpolation='bilinear')(d)  # CHECK, whats the upsample size??
        if filters2:
            d = Conv2D(filters2, kernel_size=3, strides=strides2, padding='same', activation='relu')(
                d)
            d = BatchNormalization()(d)
        return d

    def define_generator(self):
        inp = Input(shape=self.input_shape)

        d1 = self.conv2d_layer(inp, self.filters, kernel_size=7, avg_pool=False)
        d2 = self.conv2d_layer(d1, self.filters * 2, avg_pool=False, strides=2)
        d3 = self.conv2d_layer(d2, self.filters * 4, avg_pool=False, strides=2)
        d4 = self.conv2d_layer(d3, self.filters * 8)
        d5 = self.conv2d_layer(d4, self.filters * 16)
        d6 = self.conv2d_layer(d5, self.filters * 32)
        d7 = self.conv2d_layer(d6, self.filters * 32)

        d8 = self.conv2d_layer(d7, self.filters * 32)
        b0 = AvgPool2D(pool_size=(2, 2))(d8)  # CHECK ME!!
        b1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(b0)
        b2 = Conv2D(filters=self.filters * 32, kernel_size=self.kernel_size, padding='same', activation='relu')(b1)
        b3 = BatchNormalization()(b2)  # ??
        u1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(b3)
        # u1 = self.deconv2d_layer(b1, filters1=self.filters * 32, use_skip_input=False)

        u2 = self.deconv2d_layer(u1, skip_input=d7, filters1=self.filters * 64, filters2=self.filters * 32)
        u3 = self.deconv2d_layer(u2, skip_input=d6, filters1=self.filters * 64, filters2=self.filters * 16)
        u4 = self.deconv2d_layer(u3, skip_input=d5, filters1=self.filters * 32, filters2=self.filters * 8)
        u5 = self.deconv2d_layer(u4, skip_input=d4, filters1=self.filters * 16, filters2=self.filters * 4)
        u6 = self.deconv2d_layer(u5, skip_input=d3, filters1=self.filters * 8, filters2=self.filters * 4, strides2=2)

        u7 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=2, padding='same')(u6)
        u8 = BatchNormalization()(u7)
        u9 = Conv2D(filters=self.filters, kernel_size=7, padding='same')(u8)
        u10 = BatchNormalization()(u9)

        rgb_out = Conv2D(filters=3, kernel_size=7, activation='tanh')(u10)

        seg_out = Conv2D(filters=5, kernel_size=7, activation='tanh')(u10)

        depth_out = Conv2D(filters=1, kernel_size=7, activation='tanh')(u10)

        model = Model(inp, [rgb_out, seg_out, depth_out])
        print(model.summary())
        return model

    # should be multi-scale patch-based design
    # loss is a sum of discriminator loss with diff patch sizes
    def define_discriminator(self):

        img_A = Input(shape=self.input_shape)
        img_B = Input(shape=self.rgb_shape)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        # activation functions /normalisation etc???
        d1 = Conv2D(filters=self.filters, kernel_size=3)(combined_imgs)
        d2 = Conv2D(filters=self.filters, kernel_size=3)(d1)
        d3 = Conv2D(filters=self.filters * 2, kernel_size=3)(d2)
        d4 = Conv2D(filters=self.filters * 4, kernel_size=3, activation='sigmoid')(d3)  # activation???

        # NEEDS OUTPUT LAYER??

        model = Model([img_A, img_B], d4)

        print(model.summary())

        # What should loss be?? think binary crossentropy is ok
        model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])

        return model

    def define_gan(self):
        img_A = Input(shape=self.rgb_shape)
        target_RGB = Input(shape=self.rgb_shape)
        target_seg = Input(shape=self.seg_shape)
        target_depth = Input(shape=self.depth_shape)

        # fake_imgs = self.g([target_RGB, target_seg, target_depth])
        fake_RGB, fake_seg, fake_depth = self.g(img_A)

        self.d.trainable = False

        valids = self.d([fake_RGB, img_A])

        # should create valid output and recreate A
        gan = Model(inputs=[img_A, target_RGB], outputs=[valids, fake_RGB])

        # INCLUDE
        # visual equality between generated xa and target xc
        # semantic equality between mc and ma
        # depth equality between dc and da
        # use MPSE with l2 for semantic and depth auxiliary losses
        # plus sigmoid cross-entropy gan loss
        gan.compile(loss=['binary_crossentropy'], loss_weights=[], optimizer=self.opt)
        return gan

    def train(self):
        start_time = datetime.datetime.now()

        valid_patch_outputs = [np.ones((self.batch_size,) + (disc_patch, disc_patch, 1)) for disc_patch in
                               self.patch_sizes]
        fake_patch_outputs = [np.zeros((self.batch_size,) + (disc_patch, disc_patch, 1)) for disc_patch in
                              self.patch_sizes]
        # valid_outputs = np.ones((self.batch_size,) + disc_patch)
        # fake_outputs = -np.ones((self.batch_size,) + disc_patch)

        for epoch in range(self.epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(self.batch_size)):
                # train discriminator

                fake_A = self.g.predict(imgs_B)

                d_loss_real = self.d.train_on_batch([imgs_A, imgs_B], valid_patch_outputs)
                print("d loss shape: ", d_loss_real.shape)
                d_loss_fake = self.d.train_on_batch([fake_A, imgs_B], fake_patch_outputs)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train generator
                g_loss = self.g.train_on_batch([imgs_A, imgs_B], [valid_patch_outputs, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch,
                                                                                                      self.epochs,
                                                                                                      batch_i,
                                                                                                      self.data_loader.n_batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                if batch_i % self.sample_interval == 0:
                    self.sample_images()
                if epoch % self.model_save_interval == 0 and batch_i == 0:
                    self.g.save("models/model%d_%d" % (epoch, batch_i))

    def sample_images(self):
        pass


if __name__ == '__main__':
    rcan = RCAN()
    rcan.train()
