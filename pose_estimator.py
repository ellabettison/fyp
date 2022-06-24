from configparser import Interpolation
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.compat.v1.keras import backend
import tensorflow as tf

from data_loader import DataLoader
from scene_gen_callback import SceneGenCallback

from tensorflow import compat
from tensorflow import config
from tensorflow.keras.models import load_model

gpu_options = compat.v1.GPUOptions(allow_growth=True)
sess = compat.v1.Session(config=compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))


class ImageToLabel:

    def __init__(self, config):
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.repetitions=config.repetitions

        self.channels = 0
        if config.use_rand:
            self.channels += 3
        if config.use_canon:
            self.channels += 3
        if config.use_depth:
            self.channels += 1
        if config.use_seg:
            self.channels += 1
        if config.use_keypoint:
            self.channels += 3
        if config.use_2d_keypoints:
            self.channels += 1

        self.use_keypoints = config.use_keypoint

        self.use_vector = config.use_vector

        self.input_shape = (config.img_size, config.img_size, self.channels)
        self.conv_layers = config.conv_layers
        self.initialiser = initializers.RandomNormal(stddev=0.02)
        self.opt = Adam()
        self.epochs = config.epochs
        self.cnn = self.define_cnn()
        self.dataset_name = config.dataset_name

        # use 80% of datset for training, 20% for testing
        self.data_loader = DataLoader(config=config, end=config.dataset_size*0.8)
        self.test_data_loader = DataLoader(config=config, start=config.dataset_size*0.8)

        self.data_loader.batch_size = config.batch_size
        self.model_save_interval = 1
        self.sample_interval=config.sample_interval
        
        self.graph_name = config.graph_name
        self.results_folder=config.results_folder

    def conv2d_layer(self, layer_inp, filters, stride=1):
        c = Conv2D(filters, kernel_size=self.kernel_size, strides=stride, kernel_initializer=self.initialiser)(layer_inp)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        c = MaxPooling2D()(c)
        return c

    def define_cnn(self):
        inputs = {}
        
        # 
        if self.channels > 0:
            inp = Input(shape=self.input_shape)
            
            c = self.conv2d_layer(inp, self.filters, stride=1)
            for i in range(self.conv_layers):
                c = self.conv2d_layer(c, self.filters * 2 ** (i+1))
            c = Conv2D(self.filters*2** (self.conv_layers+1), kernel_size=self.kernel_size , activation='relu', 
                            kernel_initializer=self.initialiser)(c)

            c = Flatten()(c)

        if self.use_vector:
            v_inp = Input(shape=(4,4,512))
            v = Conv2D(filters=self.filters*2**5, kernel_size=(2,2), kernel_initializer=self.initialiser, activation='relu')(v_inp)
            v = BatchNormalization()(v)
            v = Conv2D(filters=self.filters*2**4, kernel_size=(2,2), kernel_initializer=self.initialiser, activation='relu')(v)
            v = BatchNormalization()(v)

            v = Flatten()(v)
            v = Dense(self.filters*2**6, activation='relu', kernel_initializer=self.initialiser)(v)
            v = Dense(self.filters*2**5, activation='relu', kernel_initializer=self.initialiser)(v)
            v = Dense(self.filters*2**4, activation='relu', kernel_initializer=self.initialiser)(v)
            

            if self.channels > 0:
                c = Concatenate()([c, v])
            else:
                c = v

        c = Dense(self.filters*2**4, activation='relu', kernel_initializer=self.initialiser)(c)
        c = Dense(3, activation='tanh', kernel_initializer=self.initialiser)(c) * 2

        # two inputs: rgb images concatenated by channels, vector inputs
        if self.use_vector:
            inputs['vector'] = v_inp
        if self.channels > 0:
            inputs['img'] = inp

        model = Model(inputs, c)

        print(model.summary())

        model.compile(loss='mse', optimizer=self.opt, metrics=[MeanAbsoluteError()])

        return model

    # reset network to train network from scratch
    def reinitialize(self, model):
        for l in model.layers:
            if hasattr(l,"kernel_initializer"):
                weights, biases = l.get_weights()
                l.set_weights([l.kernel_initializer((tf.shape(weights))),
                np.zeros(tf.shape(biases))
                ])

    def evaluate(self, model_to_load):
        cnn = load_model(model_to_load)
        cnn.evaluate(self.data_loader)

    def train(self):
        start_time = datetime.datetime.now()
        
        min_losses= np.empty(self.repetitions)
        min_val_losses = np.empty(self.repetitions)

        losses = np.empty((self.repetitions, self.epochs))
        val_losses = np.empty((self.repetitions, self.epochs))

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        
        best_performance=1
        
        for i in range(self.repetitions):
            # only save best network
            model_checkpoint_callback = ModelCheckpoint(
                filepath=("%s/models/%s_model" % (self.results_folder, self.graph_name)),
                monitor="val_loss",
                save_best_only=True,
                initial_value_threshold=best_performance
            )
            
            self.opt = Adam()
            self.cnn = self.define_cnn()
            
            history = self.cnn.fit(x=self.data_loader, validation_data=self.test_data_loader, 
                                    epochs=self.epochs, callbacks=[early_stopping_callback,
                                    model_checkpoint_callback])

            min_val_loss = min(history.history['val_loss'])

            min_losses[i] = min(history.history['loss'])
            min_val_losses[i] = min_val_loss

            if min_val_loss < best_performance:
                best_performance = min_val_loss

            losses[i, len(history.history['loss']):] = min_losses[i]
            losses[i, :len(history.history['loss'])] = history.history['loss']

            val_losses[i, len(history.history['val_loss']):] = min_val_losses[i]
            val_losses[i, :len(history.history['val_loss'])] = history.history['val_loss']
        total_time = datetime.datetime.now() - start_time


        plt.plot(np.mean(losses, axis=0))
        plt.plot(np.mean(val_losses, axis=0))
        plt.title('Mean model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("%s/%s_mean"%(self.results_folder, self.graph_name))
        plt.cla()
        plt.clf()
        plt.close()

        plt.plot(np.median(losses, axis=0))
        plt.plot(np.median(val_losses, axis=0))
        plt.title('Median model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("%s/%s_median"%(self.results_folder,self.graph_name))

        with open('%s/%s.txt'%(self.results_folder, self.graph_name), 'w') as f:
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
                    "\nmin min val loss: " +np.array2string(np.min(min_val_losses))+
                    "\ntrain time: "+str(total_time))

