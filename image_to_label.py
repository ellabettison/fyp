import datetime

from skimage.transform import resize
from tensorflow.keras.layers import Input, UpSampling1D, ReLU, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, \
    AvgPool2D, \
    UpSampling2D, Concatenate, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.activations import tanh
from tensorflow.keras import initializers
from tensorflow.keras.metrics import MeanAbsoluteError
import numpy as np

from data_loader import DataLoader
from scene_gen_callback import SceneGenCallback

from tensorflow import compat
from tensorflow import config

gpu_options = compat.v1.GPUOptions(allow_growth=True)
sess = compat.v1.Session(config=compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))


class ImageToLabel:

    def __init__(self, config):
        self.filters = 4
        self.kernel_size = 3
        self.input_shape = (config.img_size, config.img_size, 3)
        self.conv_layers = 2
        self.initialiser = initializers.RandomNormal(stddev=0.02)
        self.opt = Adam()
        self.epochs = config.epochs
        self.cnn = self.define_cnn()
        self.dataset_name = "/vol/bitbucket/efb4518/fyp/fyp/generated_imgs_numpy"
        # self.data_file_name = "RGB_peg_top_coords_in_image_array.npy"
        self.data_loader = DataLoader(config=config)
        self.test_data_loader = DataLoader(config=config)

        print("data loaders: ", self.data_loader, self.test_data_loader)
        self.data_loader.dataset_size=4500
        self.data_loader.batch_size = 16
        self.model_save_interval = 1
        self.sample_interval=200
        

    def conv2d_layer(self, layer_inp, filters, stride=1):
        c = Conv2D(filters, kernel_size=self.kernel_size, strides=stride, kernel_initializer=self.initialiser)(layer_inp)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        # c = Dropout(0.2)(c)
        c = MaxPooling2D()(c)
        return c

    def define_cnn(self):
        
        inp = Input(shape=self.input_shape)  # resize from img shape to rgb shape?

        c = self.conv2d_layer(inp, self.filters, stride=1)
        for i in range(self.conv_layers):
            c = self.conv2d_layer(c, self.filters * 2 ** (i+1))
        c = Conv2D(self.filters*2** (self.conv_layers+1), kernel_size=self.kernel_size , activation='relu', 
                        kernel_initializer=self.initialiser)(c)

        c = Flatten()(c)
        # c = Dense(32, activation='relu')(c)
        c = Dense(3, activation='tanh')(c) * 2

        model = Model(inp, c)

        print(model.summary())

        # What should loss be?? think binary crossentropy is ok
        model.compile(loss='mse', optimizer=self.opt, metrics=[MeanAbsoluteError()])

        return model

    def train(self):
        start_time = datetime.datetime.now()

        
        # test_data_loader.dataset_size = 1200
        # test_data_generator = test_data_loader.load_batch_with_data(128)

        # test_data = []
        # test_vals = []
        # for _ in range(8):
        #     (imgs_A, target_vals) = next(test_data_generator)
        #     test_data.append(imgs_A)
        #     test_vals.append(target_vals)

        # print(np.array(test_data).shape)
        # print(np.array(test_vals).shape)

        # test_data = np.reshape(np.array(test_data), (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        # test_vals = np.reshape(np.array(test_vals), (-1, 3))

        test_losses= []
        test_epochs = []

        history = self.cnn.fit(x=self.data_loader, validation_data=self.test_data_loader, 
                                epochs=self.epochs
                                , callbacks=[SceneGenCallback(sample_interval=self.sample_interval)])
                                # ,use_multiprocessing=True, workers=2)

        # for epoch in range(self.epochs):
        #     for batch_i, (imgs_A, target_vals) in enumerate(self.data_loader.load_batch_with_data(self.batch_size)):
        #         # train discriminator

        #         # print(target_vals)

        #         # print(target_vals)

        #         d_loss = self.cnn.train_on_batch(imgs_A, target_vals)

        #         elapsed_time = datetime.datetime.now() - start_time
        #         # Plot the progress
        #         print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] time: %s" % (epoch,
        #                                                                                  self.epochs,
        #                                                                                  batch_i,
        #                                                                                  self.data_loader.n_batches,
        #                                                                                  d_loss[0],
        #                                                                                  100 * d_loss[1],
        #                                                                                  elapsed_time))

        #         if batch_i % self.sample_interval == 0:
        #             # plt.imshow(imgs_A[0])
        #             # d_output = self.cnn(imgs_A.reshape(-1,256,256,3))[0]
        #             # # print("d output before: ", d_output)
        #             # d_output = self.image_to_screen(d_output[0], d_output[1], d_output[2])
        #             # # print(d_output)
        #             # # print("target vals before: ", target_vals[0])
        #             # target_vals = self.image_to_screen(target_vals[0][0], target_vals[0][1], target_vals[0][2])
        #             # # print(target_vals)
        #             # # print("Target vals: ", target_vals)
        #             # # print("d output: ", d_output)

        #             # plt.plot(d_output[0], d_output[1], 'ro')
        #             # plt.plot(target_vals[0], target_vals[1], 'go')
        #             # plt.savefig("/vol/bitbucket/efb4518/fyp/fyp/models/fig_%d_%d" % (epoch, batch_i))
        #             # # plt.show()
        #             # plt.clf()

        #             eval_results = self.cnn.evaluate(test_data, test_vals)
        #             print(" \n EVAL RESULTS: \n    ", eval_results)
        #             test_losses.append(eval_results[0])
        #             # print("trainable: ", self.cnn.get_weights())

        #         # if epoch % self.model_save_interval == 0 and batch_i == 0:
        #         #     self.cnn.save("/vol/bitbucket/efb4518/fyp/fyp/models/model%d_%d" % (epoch, batch_i))

        print(history)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig("/vol/bitbucket/efb4518/fyp/fyp/models/train_curve")

# if __name__ == "__main__":
#     cnn = ImageToLabel()
#     cnn.train(model = None)
