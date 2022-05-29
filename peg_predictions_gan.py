import datetime

import numpy as np
from tensorflow.keras.layers import Input, ReLU, Conv2D, BatchNormalization, \
    AvgPool2D, \
    UpSampling2D, Concatenate, Flatten, Dense

from tensorflow_addons.layers import InstanceNormalization # TODO: tensorflow_addons check version compatability

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import compat

from data_loader import DataLoader
import os
import matplotlib.pyplot as plt
from tensorflow import config
from tensorflow.keras import optimizers
from tensorflow.keras import initializers

gpu_options = compat.v1.GPUOptions(allow_growth=True)
sess = compat.v1.Session(config=compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))

class RCAN:

    def __init__(self, config, data_loader):
        self.dataset_name = "/vol/bitbucket/efb4518/fyp/fyp/generated_imgs_numpy"
        self.output_loc = "/vol/bitbucket/efb4518/fyp/fyp/generated_samples2"
        self.kernel_size = config.kernel_size
        self.filters = config.filters
        self.img_size = config.img_size 
        self.input_shape = (self.img_size, self.img_size, 3)
        self.rgb_shape = (self.img_size, self.img_size, 3)
        self.seg_shape = (self.img_size, self.img_size, 1)
        self.depth_shape = (self.img_size, self.img_size, 1)
        self.batch_size = config.batch_size #12
        self.patch_sizes = [self.img_size, self.img_size//2, self.img_size//4]
        self.patch_weights = [1, 1, 1]
        self.loss_weights = config.loss_weights
        self.epochs = config.epochs  # ??
        self.g_lr = config.learning_rate
        self.d_lr = config.learning_rate/2
        self.pool_size = 2
        self.initialiser_stddev = 0.02
        self.initialiser = initializers.RandomNormal(stddev=self.initialiser_stddev)
        self.use_rgb = True
        self.use_depth = config.use_depth
        self.use_seg = config.use_seg
        self.vector_repr_size = 16
        self.use_vector = config.use_vector

        self.sample_interval = config.sample_interval
        self.model_save_interval = 100

        self.g_lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.g_lr,
            decay_steps=50,
            decay_rate=1#decay_rate=0.95
        )

        self.d_lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.d_lr,
            decay_steps=50,
            decay_rate=1#decay_rate=0.95
        )

        self.opt_G = optimizers.Adam(learning_rate=self.g_lr_schedule, beta_1=0.5)  # OPTIMISER??
        self.opt_D = optimizers.Adam(learning_rate=self.d_lr_schedule, beta_1=0.5)
        self.data_loader = data_loader

        self.g = self.define_generator()
        self.d = self.define_discriminator()
        self.gan = self.define_gan()

    def conv2d_layer(self, layer_inp, filters, strides=1, avg_pool=True, kernel_size=3):
        c = layer_inp

        c = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=self.initialiser)(c)
        if avg_pool:
            c = AvgPool2D(pool_size=(2, 2))(c)
        c = InstanceNormalization()(c)
        c = ReLU()(c)

        return c

    def deconv2d_layer(self, layer_input, filters1, filters2=None, skip_input=None, use_skip_input=True,
                       upsample_size=2, strides=1,
                       strides2=1, use_upsampling=True):
        # d = layer_input
        if use_skip_input:
            layer_input = Concatenate()([layer_input, skip_input])

        d = Conv2D(filters1, kernel_size=1, strides=strides, padding='same', activation='relu', kernel_initializer=self.initialiser)(
            layer_input)
        d = InstanceNormalization()(d)

        if use_upsampling:
            d = UpSampling2D(size=upsample_size, interpolation='bilinear')(d)
        if filters2:
            d = Conv2D(filters2, kernel_size=3, strides=strides2, padding='same', activation='relu', kernel_initializer=self.initialiser)(
                d)
            d = InstanceNormalization()(d)
        return d

    def define_generator(self):

        inp = Input(shape=self.input_shape)

        d1 = self.conv2d_layer(inp, self.filters, kernel_size=7, avg_pool=False)
        d2 = self.conv2d_layer(d1, self.filters * 2, avg_pool=False, strides=2)
        d3 = self.conv2d_layer(d2, self.filters * 4, avg_pool=False, strides=2)
        d4 = self.conv2d_layer(d3, self.filters * 8)#, avg_pool=False)
        d5 = self.conv2d_layer(d4, self.filters * 16) #, avg_pool=False)
        d6 = self.conv2d_layer(d5, self.filters * 32) #, avg_pool=False)
        # d7 = self.conv2d_layer(d6, self.filters * 32)

        # d8 = Conv2D(filters=self.filters*32, kernel_size=self.kernel_size, padding='same', activation='relu', kernel_initializer=self.initialiser)(d5)
        b1 = self.conv2d_layer(d6, self.filters * 32, avg_pool=False)
        # b2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(b1)
        b3 = Conv2D(filters=self.filters * 32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=self.initialiser)(b1)
        # b4 = UpSampling2D(size=(2,2), interpolation='bilinear')(b3)

        # u1 = self.deconv2d_layer(d8, filters1=self.filters * 32, use_skip_input=False, use_upsampling=False)

        # u2 = self.deconv2d_layer(u1, skip_input=d6, filters1=self.filters * 64, filters2=self.filters * 32)#, use_upsampling=False)
        # u2 = self.deconv2d_layer(b3, skip_input=d7, filters1=self.filters * 64, filters2=self.filters * 32)
        u3 = self.deconv2d_layer(b3, skip_input=d6, filters1=self.filters * 64, filters2=self.filters * 16) #, use_upsampling=False)
        u4 = self.deconv2d_layer(u3, skip_input=d5, filters1=self.filters * 32, filters2=self.filters * 8)#, use_upsampling=False)
        u5 = self.deconv2d_layer(u4, skip_input=d4, filters1=self.filters * 16, filters2=self.filters * 4)#, use_upsampling=False)
        u6 = self.deconv2d_layer(u5, skip_input=d3, filters1=self.filters * 8, filters2=self.filters * 4) #, use_upsampling=False)
        
        u7 = UpSampling2D(size=(2,2), interpolation='bilinear')(u6)
        u8 = self.deconv2d_layer(u7, use_skip_input=False, filters1=self.filters * 4, filters2=self.filters *2, strides2=2)
        u9 = Conv2D(filters=self.filters * 2, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=self.initialiser)(u8)
        u10 = InstanceNormalization()(u9)

        u11 = UpSampling2D(size=(2,2), interpolation='bilinear')(u10)

        u12 = Conv2D(filters=self.filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer=self.initialiser)(u11)
        u13 = InstanceNormalization()(u12)
 
        # u10 = Conv2D(filters=self.filters, kernel_size=3, padding='same')(u9)
        # u11 = InstanceNormalization()(u10)

        # TODO: REMOVED TANH ACTIVATION TEMPORARILY
        # u_rgb = Conv2D(filters=self.filters * 2, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=self.initialiser)(u8)
        # u_rgb_2 = InstanceNormalization()(u_rgb)
        

        # seg_out = Conv2D(filters=5, kernel_size=7, activation='tanh')(u10)
        
        # TODO: REMOVED SIGMOID ACTIVATION TEMPORARILY
        # u_seg = Conv2D(filters=self.filters * 2, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=self.initialiser)(u8)
        # u_seg_2 = InstanceNormalization()(u_seg)

        gen_outputs = {}

        if self.use_rgb:
            rgb_out = Conv2D(filters=3, kernel_size=7, padding='same', activation='tanh', kernel_initializer=self.initialiser)(u13)
            gen_outputs['fake_rgb'] = rgb_out

        if self.use_seg:
            seg_out = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer=self.initialiser)(u13)
            gen_outputs['fake_seg'] = seg_out

        if self.use_depth:
            depth_out = Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer=self.initialiser)(u13)
            gen_outputs['fake_depth'] = depth_out

        if self.use_vector:
            u14 = Flatten()(u13)
            vector_repr = Dense(16)(u14)
            gen_outputs['vector_repr'] = vector_repr
            

        model = Model(inp, gen_outputs)

        print(model.summary())
        
        return model

    # should be multi-scale patch-based design
    # loss is a sum of discriminator loss with diff patch sizes
    def define_discriminator(self):

        img_A = Input(shape=self.input_shape)  # resize from img shape to rgb shape?
        cond_img = Input(shape=self.input_shape)

        combined_imgs = Concatenate(axis=-1)([img_A, cond_img])

        combined_imgs_downsample1 = AvgPool2D()(combined_imgs)

        combined_imgs_downsample2 = AvgPool2D()(combined_imgs_downsample1)

        # sub discrim 1
        d1 = self.conv2d_layer(combined_imgs, filters=self.filters, kernel_size=3, avg_pool=False)
        d2 = self.conv2d_layer(d1, filters=self.filters, kernel_size=3, avg_pool=False)
        d3 = self.conv2d_layer(d2, filters=self.filters * 2, kernel_size=3, avg_pool=False)
        d4 = self.conv2d_layer(d3, filters=self.filters * 4, kernel_size=3, avg_pool=False)
        o1 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', kernel_initializer=self.initialiser)(d4)

        # sub discrim 2
        e1 = self.conv2d_layer(combined_imgs_downsample1, filters=self.filters, kernel_size=3, avg_pool=False)
        e2 = self.conv2d_layer(e1, filters=self.filters, kernel_size=3, avg_pool=False)
        e3 = self.conv2d_layer(e2, filters=self.filters * 2, kernel_size=3, avg_pool=False)
        e4 = self.conv2d_layer(e3, filters=self.filters * 4, kernel_size=3, avg_pool=False)
        o2 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', kernel_initializer=self.initialiser)(e4) 

        # sub discrim 3
        f1 = self.conv2d_layer(combined_imgs_downsample2, filters=self.filters, kernel_size=3, avg_pool=False)
        f2 = self.conv2d_layer(f1, filters=self.filters, kernel_size=3, avg_pool=False)
        f3 = self.conv2d_layer(f2, filters=self.filters * 2, kernel_size=3, avg_pool=False)
        f4 = self.conv2d_layer(f3, filters=self.filters * 4, kernel_size=3, avg_pool=False)
        o3 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', kernel_initializer=self.initialiser)(f4) 

        model = Model(inputs=[img_A, cond_img], outputs=[o1, o2, o3])

        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer=self.opt_D) 

        return model

    def define_gan(self):
        img_A = Input(shape=self.input_shape)

        fake_imgs = self.g(img_A)

        self.d.trainable = False

        valids = self.d([fake_imgs['fake_rgb'], img_A])

        print("shapes: ", fake_imgs.keys())

        # should create valid output and recreate A
        outputs = {"valids_0": valids[0], "valids_1":valids[1], "valids_2":valids[2], "fake_rgb":fake_imgs['fake_rgb']}
        
        # INCLUDE
        # visual equality between generated xa and target xc
        # semantic equality between mc and ma
        # depth equality between dc and da
        # use MPSE with l2 for semantic and depth auxiliary losses
        # plus sigmoid cross-entropy gan loss
        # instead of mse, should be compat.v1.losses.mean_pairwise_squared_error
        losses = {"valids_0":'binary_crossentropy', "valids_1":'binary_crossentropy', 'valids_2':'binary_crossentropy', 'fake_rgb':'mse'}
        loss_weights = {"valids_0":self.loss_weights[0],"valids_1":self.loss_weights[1],"valids_2":self.loss_weights[2],'fake_rgb':self.loss_weights[3]}

        if self.use_seg:
            losses["fake_seg"] = "mse"
            loss_weights["fake_seg"] = self.loss_weights[4]
            outputs["fake_seg"] = fake_imgs['fake_seg']
        
        if self.use_depth:
            losses['fake_depth'] = 'mse'
            loss_weights['fake_depth'] = self.loss_weights[3]
            outputs['fake_depth'] = fake_imgs['fake_depth']

        gan = Model(inputs=img_A, outputs=outputs)
        gan.compile(loss=losses, loss_weights=loss_weights, optimizer=self.opt_G)
        return gan

    def train(self):
        start_time = datetime.datetime.now()

        # valid_patch_outputs = [np.ones((self.batch_size,) + (disc_patch, disc_patch, 1)) for disc_patch in
                            #    self.patch_sizes]
        valid_patch_outputs = [np.full((self.batch_size,) + (disc_patch, disc_patch, 1), 0.9) for disc_patch in
                               self.patch_sizes]
        # fake_patch_outputs = [np.zeros((self.batch_size,) + (disc_patch, disc_patch, 1)) for disc_patch in
                            #   self.patch_sizes]
        fake_patch_outputs = [np.full((self.batch_size,) + (disc_patch, disc_patch, 1), 0.1) for disc_patch in
                              self.patch_sizes]


        for epoch in range(self.epochs):
            for batch_i, (target_img, randomised_img, target_depth, target_segmap) in enumerate(self.data_loader.load_batch(self.batch_size)):
                # train discriminator

                fake_imgs = self.g.predict(randomised_img)

                print("ERROR: ", np.sum(np.abs(np.subtract(target_img, fake_imgs['fake_rgb']))))

                d_loss_real = self.d.train_on_batch([target_img, randomised_img], valid_patch_outputs)

                d_loss_fake = self.d.train_on_batch([fake_imgs['fake_rgb'], randomised_img], fake_patch_outputs)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                gan_outputs = {'valids_0':valid_patch_outputs[0], 'valids_1':valid_patch_outputs[1], 'valids_2':valid_patch_outputs[2], 'fake_rgb':target_img}

                if self.use_seg:
                    gan_outputs['fake_seg'] = target_segmap
                
                if self.use_depth:
                    gan_outputs['fake_depth'] = target_depth

                # train generator

                g_loss = self.gan.train_on_batch(randomised_img, gan_outputs)

                print("GAN LOSS: ", g_loss)
                print("DIM LOSS: ", d_loss)

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

                print("LEARNIGN RATE: ", self.opt_G._decayed_lr('float32'))

                sample_imgs_config = {
                    'epoch':epoch, 
                    'batch_i':batch_i, 
                    'randomised_imgs':randomised_img[:3], 
                    'target_imgs':target_img[:3], 
                    'fake_imgs':fake_imgs['fake_rgb'][:3],  
                }

                if self.use_seg:
                    sample_imgs_config['target_segmaps'] = target_segmap[:3]
                    sample_imgs_config['fake_segs'] = fake_imgs['fake_seg'][:3]

                if self.use_depth:
                    sample_imgs_config['target_depths'] = target_depth[:3], 
                    sample_imgs_config['fake_depths'] = fake_imgs['fake_depth'][:3]


                if batch_i % self.sample_interval == 0:
                    self.sample_images(**sample_imgs_config)
                if epoch % self.model_save_interval == 0 and batch_i == 0:
                    self.g.save("models/model%d_%d" % (epoch, batch_i))

    def sample_images(self, epoch, batch_i, randomised_imgs, target_imgs, fake_imgs, target_segmaps=[], fake_segs=[], target_depths=[], fake_depths=[]):
        # print("saving figs")
        os.makedirs('%s' % (self.output_loc), exist_ok=True)
        r, c = 3, 3

        # target_img, randomised_img, _, target_segmap = self.data_loader.load_data(batch_size=4)
        # fake_A = self.g.predict(randomised_img)

        # print("\nTARGET IMG: ", target_imgs[0])
        # print("\nFAKE IMG: ", fake_imgs[0])

        fake_seg = np.repeat(fake_segs, 3, axis=-1) # -0.5) * 2 # TODO: why is this darker?? removed the 0.5 for now but not good solution!!

        if self.use_seg:
            target_segmap = (np.repeat(target_segmaps, 3, axis=-1) - 0.5) * 2
            seg_imgs = np.concatenate([randomised_imgs, fake_seg, target_segmap])
            seg_imgs = 0.5 * seg_imgs + 0.5

        if self.use_depth:
            target_depth = (np.repeat(target_depths, 3, axis=-1) - 0.5) * 2
            depth_imgs = np.concatenate([randomised_imgs, fake_depths, target_depth])
            depth_imgs = 0.5 * depth_imgs + 0.5

        gen_imgs = np.concatenate([randomised_imgs, fake_imgs, target_imgs])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        gen_cnt = 0
        seg_cnt = 0
        depth_cnt = 0

        for i in range(r):
            # for j in range(c):
            axs[i, 0].imshow(gen_imgs[gen_cnt])
            axs[i, 0].set_title(titles[i])
            axs[i, 0].axis('off')
            gen_cnt += 1

            axs[i, 1].imshow(gen_imgs[gen_cnt])
            axs[i, 1].set_title(titles[i])
            axs[i, 1].axis('off')
            gen_cnt += 1

            axs[i, 2].imshow(gen_imgs[gen_cnt])
            axs[i, 2].set_title(titles[i])
            axs[i, 2].axis('off')
            gen_cnt += 1

            # if self.use_seg:
            #     axs[i, 1].imshow(seg_imgs[seg_cnt])
            #     seg_cnt += 1
            # # else:
            # #     axs[i, 1].imshow(gen_imgs[gen_cnt])
            # #     gen_cnt += 1
            # axs[i, 1].set_title(titles[i])
            # axs[i, 1].axis('off')

            # if self.use_depth:
            #     axs[i, 2].imshow(depth_imgs[depth_cnt])
            #     depth_cnt += 1
            # # else:
            # #     axs[i, 2].imshow(gen_imgs[gen_cnt])
            # #     gen_cnt += 1
            # axs[i, 2].set_title(titles[i])
            # axs[i, 2].axis('off')

            # axs[i, 3].imshow(gen_imgs[cnt])
            # axs[i, 3].set_title(titles[i])
            # axs[i, 3].axis('off')
            # cnt += 1

        fig.savefig("%s/%d_%d.png" % (self.output_loc, epoch, batch_i))
        plt.show()
        plt.close()

