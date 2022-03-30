import datetime

import numpy as np
from tensorflow.keras.layers import Input, ReLU, Conv2D, BatchNormalization, \
    AvgPool2D, \
    UpSampling2D, Concatenate

from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import BinaryCrossentropy, mae
# from skimage.transform import resize
from tensorflow import compat

from data_loader import DataLoader
import os
import matplotlib.pyplot as plt
from tensorflow import config
from tensorflow import debugging
# from tensorflow.keras import backend as K

sess = compat.v1.Session(config=compat.v1.ConfigProto(log_device_placement=True))

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))
# debugging.set_log_device_placement(True)

# def loss_function(xs, ys):
#     print("xs shape: ", xs.shape)
#     return np.mean([tensorflow.keras.losses.binary_crossentropy(xs[i], ys[i]) for i in range(xs)])


class RCAN:

    def __init__(self):
        self.dataset_name = "output"
        self.output_loc = "generated_imgs"
        self.kernel_size = 3
        self.filters = 32
        # self.img_shape = (472, 472, 3) # from paper
        self.img_size = 64  # 512
        self.output_size = 58
        self.input_shape = (self.img_size, self.img_size, 3)
        self.rgb_shape = (self.output_size, self.output_size, 3)
        self.seg_shape = (self.output_size, self.output_size, 5)
        self.depth_shape = (self.output_size, self.output_size, 1)
        self.batch_size = 16
        # self.patch_sizes = [472, 236, 118]
        # self.patch_sizes = [512, 256, 128]
        self.patch_sizes = [64, 32, 16]
        self.patch_weights = [1, 1, 1]
        self.epochs = 20  # ??
        self.lr = None
        self.pool_size = 2

        self.sample_interval = 5
        self.model_save_interval = 100

        self.opt_G = Adam(lr=0.0002)  # OPTIMISER??
        self.opt_D = Adam(lr=0.0002)
        self.data_loader = DataLoader(dataset_name=self.dataset_name)

        self.g = self.define_generator()
        self.d = self.define_discriminator()
        self.gan = self.define_gan()

    def conv2d_layer(self, layer_inp, filters, strides=1, avg_pool=True, kernel_size=3):
        c = layer_inp

        c = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(c)
        if avg_pool:
            c = AvgPool2D(pool_size=(2, 2))(c)
        c = InstanceNormalization()(c)
        c = ReLU()(c)

        return c

    def deconv2d_layer(self, layer_input, filters1, filters2=None, skip_input=None, use_skip_input=True,
                       upsample_size=2, strides=1,
                       strides2=1):
        # d = layer_input
        if use_skip_input:
            layer_input = Concatenate()([layer_input, skip_input])

        d = Conv2D(filters1, kernel_size=1, strides=strides, padding='same', activation='relu')(
            layer_input)
        d = InstanceNormalization()(d) # change to instance norm

        d = UpSampling2D(size=upsample_size, interpolation='bilinear')(d)  # CHECK, whats the upsample size??
        if filters2:
            d = Conv2D(filters2, kernel_size=3, strides=strides2, padding='same', activation='relu')(
                d)
            d = InstanceNormalization()(d)
        return d

    def define_generator(self):
        inp = Input(shape=self.input_shape)

        d1 = self.conv2d_layer(inp, self.filters, kernel_size=7, avg_pool=False)
        d2 = self.conv2d_layer(d1, self.filters * 2, avg_pool=False, strides=2)
        d3 = self.conv2d_layer(d2, self.filters * 4, avg_pool=False, strides=2)
        d4 = self.conv2d_layer(d3, self.filters * 8)
        d5 = self.conv2d_layer(d4, self.filters * 16)
        # d6 = self.conv2d_layer(d5, self.filters * 32)
        # d7 = self.conv2d_layer(d6, self.filters * 32)

        d8 = self.conv2d_layer(d5, self.filters * 32)
        b0 = AvgPool2D(pool_size=(2, 2))(d8)  # CHECK ME!!
        b1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(b0)
        b2 = Conv2D(filters=self.filters * 32, kernel_size=self.kernel_size, padding='same', activation='relu')(b1)
        b3 = InstanceNormalization()(b2)  # ??
        u1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(b3)
        # u1 = self.deconv2d_layer(b1, filters1=self.filters * 32, use_skip_input=False)

        # u2 = self.deconv2d_layer(u1, skip_input=d7, filters1=self.filters * 64, filters2=self.filters * 32)
        # u3 = self.deconv2d_layer(u1, skip_input=d6, filters1=self.filters * 64, filters2=self.filters * 16)
        u4 = self.deconv2d_layer(u1, skip_input=d5, filters1=self.filters * 32, filters2=self.filters * 8)
        u5 = self.deconv2d_layer(u4, skip_input=d4, filters1=self.filters * 16, filters2=self.filters * 4)
        u6 = self.deconv2d_layer(u5, skip_input=d3, filters1=self.filters * 8, filters2=self.filters * 2)

        u7 = self.deconv2d_layer(u6, skip_input=d2, filters1=self.filters * 4, filters2=self.filters)

        # u8 = self.deconv2d_layer(u5, skip_input=d3, filters1=self.filters * 8, filters2=self.filters * 4)
        # u7 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=2, padding='same')(u6)
        # u8 = BatchNormalization()(u7)
        # u9 = Conv2D(filters=self.filters, kernel_size=7, padding='same')(u8)
        # u10 = BatchNormalization()(u9)

        rgb_out = Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='tanh')(u7)

        # seg_out = Conv2D(filters=5, kernel_size=7, activation='tanh')(u10)
        #
        # depth_out = Conv2D(filters=1, kernel_size=7, activation='tanh')(u10)

        model = Model(inp, rgb_out)  # , seg_out, depth_out])
        print(model.summary())
        return model

    def sub_discriminator(self, layer_input):
        pass

    # should be multi-scale patch-based design
    # loss is a sum of discriminator loss with diff patch sizes
    def define_discriminator(self):

        img_A = Input(shape=self.input_shape)  # resize from img shape to rgb shape?
        img_B = Input(shape=self.input_shape)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        combined_imgs_downsample1 = AvgPool2D()(combined_imgs)

        combined_imgs_downsample2 = AvgPool2D()(combined_imgs_downsample1)

        # sub discrim 1
        d1 = self.conv2d_layer(combined_imgs, filters=self.filters, kernel_size=3, avg_pool=False)
        d2 = self.conv2d_layer(d1, filters=self.filters, kernel_size=3, avg_pool=False)
        d3 = self.conv2d_layer(d2, filters=self.filters * 2, kernel_size=3, avg_pool=False)
        d4 = self.conv2d_layer(d3, filters=self.filters * 4, kernel_size=3, avg_pool=False)
        o1 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(d4)  # ??

        # sub discrim 2
        e1 = self.conv2d_layer(combined_imgs_downsample1, filters=self.filters, kernel_size=3, avg_pool=False)
        e2 = self.conv2d_layer(e1, filters=self.filters, kernel_size=3, avg_pool=False)
        e3 = self.conv2d_layer(e2, filters=self.filters * 2, kernel_size=3, avg_pool=False)
        e4 = self.conv2d_layer(e3, filters=self.filters * 4, kernel_size=3, avg_pool=False)
        o2 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(e4)  # ??

        # sub discrim 3
        f1 = self.conv2d_layer(combined_imgs_downsample2, filters=self.filters, kernel_size=3, avg_pool=False)
        f2 = self.conv2d_layer(f1, filters=self.filters, kernel_size=3, avg_pool=False)
        f3 = self.conv2d_layer(f2, filters=self.filters * 2, kernel_size=3, avg_pool=False)
        f4 = self.conv2d_layer(f3, filters=self.filters * 4, kernel_size=3, avg_pool=False)
        o3 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(f4)  # ??

        # o1 = sub_descrim(combined_imgs)
        # o2 = sub_descrim(combined_imgs_downsample1)
        # o3 = sub_descrim(combined_imgs_downsample2)

        model = Model([img_A, img_B], [o1, o2, o3])

        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer=self.opt_D)  # , metrics=['accuracy'])

        return model

    def define_gan(self):
        img_A = Input(shape=self.input_shape)
        # img_A_cropped = Input(shape=self.rgb_shape)
        target_RGB = Input(shape=self.input_shape)
        # target_seg = Input(shape=self.seg_shape)
        # target_depth = Input(shape=self.depth_shape)

        fake_RGB = self.g(img_A)

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
        gan.compile(loss=['binary_crossentropy', compat.v1.losses.mean_pairwise_squared_error], loss_weights=[1, 1], optimizer=self.opt_G)
        return gan

    def train(self):
        start_time = datetime.datetime.now()

        valid_patch_outputs = [np.ones((self.batch_size,) + (disc_patch, disc_patch, 1)) for disc_patch in
                               self.patch_sizes]
        # valid_patch_outputs = [np.full((self.batch_size,) + (disc_patch, disc_patch, 1), 0.9) for disc_patch in
        #                        self.patch_sizes]
        fake_patch_outputs = [np.zeros((self.batch_size,) + (disc_patch, disc_patch, 1)) for disc_patch in
                              self.patch_sizes]
        # fake_patch_outputs = [np.full((self.batch_size,) + (disc_patch, disc_patch, 1), 0.1) for disc_patch in
        #                       self.patch_sizes]

        print(valid_patch_outputs[0].shape, valid_patch_outputs[1].shape, valid_patch_outputs[2].shape)
        # valid_outputs = np.ones((self.batch_size,) + disc_patch)
        # fake_outputs = -np.ones((self.batch_size,) + disc_patch)

        for epoch in range(self.epochs):
            for batch_i, (target_img, randomised_img) in enumerate(self.data_loader.load_batch(self.batch_size)):
                # train discriminator

                fake_A = self.g.predict(randomised_img)

                print("ERROR: ", np.sum(np.abs(np.subtract(target_img, fake_A))))

                # print(imgs_A.shape, imgs_B.shape)
                # gen_imgs = 0.5 * imgs_A[0] + 0.5
                # gen_imgs2 = 0.5 * imgs_B[0] + 0.5
                # plt.imshow(gen_imgs)
                # plt.show()
                # plt.imshow(gen_imgs2)
                # plt.show()
                d_loss_real = self.d.train_on_batch([target_img, randomised_img], valid_patch_outputs)

                # d_loss_1 = self.d.train_on_batch([imgs_A, imgs_B], valid_patch_outputs[0])
                # d_loss_2 = self.d.train_on_batch([imgs_A, imgs_B], valid_patch_outputs[1])
                # d_loss_3 = self.d.train_on_batch([imgs_A, imgs_B], valid_patch_outputs[2])
                #
                # print("losses: ", d_loss_1, d_loss_2, d_loss_3)
                # print("total loss: ", d_loss_real)

                # print("d loss shape: ", d_loss_real)
                d_loss_fake = self.d.train_on_batch([fake_A, randomised_img], fake_patch_outputs)

                # print("d loss fake: ", d_loss_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # train generator
                g_loss = self.gan.train_on_batch([randomised_img, target_img], [valid_patch_outputs, target_img])

                print("GAN LOSS: ", g_loss)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" % (epoch,
                                                                                          self.epochs,
                                                                                          batch_i,
                                                                                          self.data_loader.n_batches,
                                                                                          d_loss[0],
                                                                                          # 100 * d_loss[1],
                                                                                          g_loss[0],
                                                                                          elapsed_time))

                if batch_i % self.sample_interval == 0:
                    self.sample_images()
                if epoch % self.model_save_interval == 0 and batch_i == 0:
                    self.g.save("models/model%d_%d" % (epoch, batch_i))

    def sample_images(self):
        # print("saving figs")
        os.makedirs('%s/%s' % (self.output_loc, self.dataset_name), exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3)
        fake_A = self.g.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # print(fake_A)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        # fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
        plt.show()
        plt.close()


if __name__ == '__main__':
    rcan = RCAN()
    rcan.train()
