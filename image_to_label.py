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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras import backend
import tensorflow as tf

from data_loader import DataLoader
from scene_gen_callback import SceneGenCallback

from tensorflow import compat
from tensorflow import config

gpu_options = compat.v1.GPUOptions(allow_growth=True)
sess = compat.v1.Session(config=compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))


class ImageToLabel:

    def __init__(self, config):
        self.filters = config.filters
        self.kernel_size = config.kernel_size

        channels = 0
        if config.use_rand:
            channels += 3
        if config.use_canon:
            channels += 3
        if config.use_depth:
            channels += 1
        if config.use_seg:
            channels += 1

        self.input_shape = (config.img_size, config.img_size, channels)
        self.conv_layers = 2
        self.initialiser = initializers.RandomNormal(stddev=0.02)
        self.opt = Adam()
        self.epochs = config.epochs
        self.cnn = self.define_cnn()
        self.dataset_name = "/vol/bitbucket/efb4518/fyp/fyp/generated_imgs_numpy"
        # self.data_file_name = "RGB_peg_top_coords_in_image_array.npy"
        self.data_loader = DataLoader(config=config, end=4000)
        self.test_data_loader = DataLoader(config=config, start=4000)

        # self.use_canon = True
        # self.use_depth = True
        # self.use_seg = True

        print("data loaders: ", self.data_loader, self.test_data_loader)
        # self.data_loader.dataset_size=4500
        self.data_loader.batch_size = 16
        self.model_save_interval = 1
        self.sample_interval=200
        self.repetitions=config.repetitions
        self.graph_name = config.graph_name
        

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

    def reinitialize(self, model):
        for l in model.layers:
            # session = backend.get_session()
            if hasattr(l,"kernel_initializer"):
                weights, biases = l.get_weights()
                # weights_initializer = tf.variables_initializer(l.weights)
                l.set_weights([l.kernel_initializer((tf.shape(weights))),
                np.zeros(tf.shape(biases))
                ])
            else:
                print("aaagggggaggggagag")
                # session.run(weights_initializer)
            # if hasattr(l,"bias_initializer"):
            #     l.set_bias(l.bias_initializer(tf.shape(l.bias)))
                # weights_initializer = tf.variables_initializer(l.biases)
                # session.run(bias_initializer)
            # if hasattr(l,"recurrent_initializer"):
            #     l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            # session.run(l.kernel_initializer)

    def train(self):
        start_time = datetime.datetime.now()
        
        min_losses= np.empty(self.repetitions)
        min_val_losses = np.empty(self.repetitions)

        losses = np.empty((self.repetitions, self.epochs))
        val_losses = np.empty((self.repetitions, self.epochs))

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        for i in range(self.repetitions):
            
            # backend.clear_session()
            # self.cnn.reset_states()
            # self.reinitialize(self.cnn)
            self.opt = Adam()
            self.cnn = self.define_cnn()
            
            history = self.cnn.fit(x=self.data_loader, validation_data=self.test_data_loader, 
                                    epochs=self.epochs, callbacks=[early_stopping_callback])
                                    # , callbacks=[SceneGenCallback(sample_interval=self.sample_interval)])
                                    # ,use_multiprocessing=True, workers=2)

            min_losses[i] = min(history.history['loss'])
            min_val_losses[i] = min(history.history['val_loss'])

            losses[i, len(history.history['loss']):] = min_losses[i]
            losses[i, :len(history.history['loss'])] = history.history['loss']

            val_losses[i, len(history.history['val_loss']):] = min_val_losses[i]
            val_losses[i, :len(history.history['val_loss'])] = history.history['val_loss']


        # print(history)
        plt.plot(np.mean(losses, axis=0))
        plt.plot(np.mean(val_losses, axis=0))
        plt.title('Mean model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig("/vol/bitbucket/efb4518/fyp/fyp/results/%s_mean"%self.graph_name)
        plt.cla()
        plt.clf()
        plt.close()


        plt.plot(np.median(losses, axis=0))
        plt.plot(np.median(val_losses, axis=0))
        plt.title('Median model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig("/vol/bitbucket/efb4518/fyp/fyp/results/%s_median"%self.graph_name)

        with open('/vol/bitbucket/efb4518/fyp/fyp/results/%s.txt'%self.graph_name, 'w') as f:
            f.write("losses: " +np.array2string(losses) +
                    "\nval losses: " +np.array2string(val_losses) +
                    "\nmean losses: "+ np.array2string(np.mean(losses, axis=0))+
                    "\nmean val losses: " + np.array2string(np.mean(val_losses, axis=0))+
                    "\nmedian losses: "+ np.array2string(np.median(losses, axis=0))+
                    "\nmedian val losses: " + np.array2string(np.median(val_losses, axis=0))+
                    "\nmean min loss: " + np.array2string(np.mean(min_losses))+
                    "\nmedian min loss: " + np.array2string(np.median(min_losses))+
                    "\nmin min loss: " + np.array2string(np.min(min_losses))+
                    "\nmean min val loss: " + np.array2string(np.mean(min_val_losses))+
                    "\nmedian min val loss: " + np.array2string(np.median(min_val_losses))+
                    "\nmin min val loss: " +np.array2string(np.min(min_val_losses)))


# if __name__ == "__main__":
#     cnn = ImageToLabel()
#     cnn.train(model = None)
