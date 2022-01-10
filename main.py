import datetime

import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, AvgPool2D, \
    UpSampling2D, Concatenate, Dropout
# import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.losses import BinaryCrossentropy, mae


class RCAN:

    def __init__(self):
        self.kernel_size = 3
        self.filters = 32
        self.img_shape = None
        self.batch_size = None
        self.patch_sizes = [472, 236, 118]
        self.patch_weights = [1, 1, 1]
        self.epochs = None
        self.lr = None

        self.sample_interval = 10
        self.model_save_interval = 100

        self.g = self.generator()
        self.d = None
        self.gan = None

        self.opt = None  # OPTIMISER??

    def conv2d_layer(self, layer_inp, filters, strides=1, avg_pool=True, kernel_size=3):
        c = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False,
                   activation='relu')(layer_inp)
        c = BatchNormalization(momentum=0.8)(c)  # Change to instance norm
        if avg_pool:
            c = AvgPool2D(c)  # CHECK ME!!
        return c

    def deconv2d_layer(self, layer_input, filters1, filters2=None, skip_input=None, upsample_size=1, strides=1,
                       strides2=1):
        d = Conv2D(filters1, kernel_size=1, strides=strides, padding='same', activation='relu')(
            layer_input)
        d = BatchNormalization(momentum=0.8)(d)
        if skip_input:
            d = Concatenate()([d, skip_input])
        d = UpSampling2D(size=upsample_size, interpolation='bilinear')(d)  # CHECK, whats the upsample size??
        if filters2:
            d = Conv2D(filters2, kernel_size=3, strides=strides2, padding='same', activation='relu')(
                layer_input)
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def generator(self):
        inp = Input(shape=self.img_shape)

        d1 = self.conv2d_layer(inp, self.filters, kernel_size=7, avg_pool=False)  # should this be 7??
        d2 = self.conv2d_layer(d1, self.filters * 2, avg_pool=False, strides=2)
        d3 = self.conv2d_layer(d2, self.filters * 4, avg_pool=False, strides=2)
        d4 = self.conv2d_layer(d3, self.filters * 8)
        d5 = self.conv2d_layer(d4, self.filters * 16)
        d6 = self.conv2d_layer(d5, self.filters * 32)
        d7 = self.conv2d_layer(d6, self.filters * 32)

        d8 = self.conv2d_layer(d7, self.filters * 32)
        b1 = UpSampling2D(interpolation='bilinear')(d8)
        b2 = Conv2D(filters=self.filters * 32)(b1)
        u1 = self.deconv2d_layer(b2, filters1=self.filters * 32)

        u2 = self.deconv2d_layer(u1, skip_input=d7, filters1=self.filters * 64, filters2=self.filters * 32)
        u3 = self.deconv2d_layer(u2, skip_input=d6, filters1=self.filters * 64, filters2=self.filters * 16)
        u4 = self.deconv2d_layer(u3, skip_input=d5, filters1=self.filters * 32, filters2=self.filters * 8)
        u5 = self.deconv2d_layer(u4, skip_input=d4, filters1=self.filters * 16, filters2=self.filters * 4)
        u6 = self.deconv2d_layer(u5, skip_input=d3, filters1=self.filters * 8, filters2=self.filters * 4, strides2=2)

        u7 = Conv2D(filters=self.filters * 2)(u6)
        u8 = BatchNormalization(momentum=0.8)(u7)
        u9 = Conv2D(filters=self.filters)(u8)
        u10 = BatchNormalization(momentum=0.8)(u9)

        rgb_out = Conv2D(filters=3, kernel_size=7, activation='tanh')(u10)

        seg_out = Conv2D(filters=5, kernel_size=7, activation='tanh')(u10)

        depth_out = Conv2D(filters=1, kernel_size=7, activation='tanh')(u10)

        return Model(inp, [rgb_out, seg_out, depth_out])

    # should be multi-scale patch-based design
    # loss is a sum of discriminator loss with diff patch sizes
    def discriminator(self):

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        # activation functions /normalisation etc???
        d1 = Conv2D(filters=self.filters, kernel_size=3)(combined_imgs)
        d2 = Conv2D(filters=self.filters, kernel_size=3)(d1)
        d3 = Conv2D(filters=self.filters * 2, kernel_size=3)(d2)
        d4 = Conv2D(filters=self.filters * 4, kernel_size=3)(d3)

        # NEEDS OUTPUT LAYER??

        model = Model([img_A, img_B], d4)

        # What should loss be?? think binary crossentropy is ok
        model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])

        return model

    def gan(self):
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_A = self.g(img_B)

        self.d.trainable = False

        valid = self.d([fake_A, img_B])

        # should create valid output and recreate A
        gan = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])

        # INCLUDE
        # visual equality between generated xa and target xc
        # semantic equality between mc and ma
        # depth equality between dc and da
        # use MPSE with l2 for semantic and depth auxiliary losses
        # plus sigmoid cross-entropy gan loss
        gan.compile(loss=[], loss_weights=[], optimizer=self.opt)
        return gan

    def train(self):
        start_time = datetime.datetime.now()

        valid_outputs = np.ones((self.batch_size,) + disc_patch)
        fake_outputs = -np.ones((self.batch_size,) + disc_patch)

        for epoch in range(self.epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(self.batch_size)):
                # train discriminator

                fake_A = self.g.predict(imgs_B)

                d_loss_real = self.d.train_on_batch([imgs_A, imgs_B], valid_outputs)
                d_loss_fake = self.d.train_on_batch([fake_A, imgs_B], fake_outputs)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train generator
                g_loss = self.g.train_on_batch([imgs_A, imgs_B], [valid_outputs, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                      batch_i,
                                                                                                      data_loader.n_batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                if batch_i % self.sample_interval == 0:
                    self.sample_images()
                if epoch % self.model_save_interval == 0 and batch_i == 0:
                    g_model.save("models/model%d_%d" % (epoch, batch_i))

    def sample_images(self):
        pass


if __name__ == '__main__':
    rcan = RCAN()
    rcan.train()
