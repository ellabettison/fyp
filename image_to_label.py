import datetime

from skimage.transform import resize
from tensorflow.keras.layers import Input, UpSampling1D, ReLU, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, \
    AvgPool2D, \
    UpSampling2D, Concatenate, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from data_loader import DataLoader


class ImageToLabel:

    def __init__(self):
        self.filters = 8
        self.kernel_size = 2
        self.input_shape = (72, 128, 3)
        self.conv_layers = 1
        self.opt = Adam()
        self.epochs = 100
        self.cnn = self.define_cnn()
        self.dataset_name = "peg_predictions_dataset"
        self.data_file_name = "RGB_peg_top_coords_in_image_array.npy"
        self.data_loader = DataLoader(dataset_name=self.dataset_name, data_to_use=self.data_file_name)
        self.batch_size = 10
        self.model_save_interval = 1
        self.sample_interval=100

    def conv2d_layer(self, layer_inp, filters):
        c = Conv2D(filters, kernel_size=self.kernel_size, padding='same')(layer_inp)
        c = BatchNormalization()(c)
        c = ReLU()(c)
        return c

    def define_cnn(self):
        inp = Input(shape=self.input_shape)  # resize from img shape to rgb shape?

        c = self.conv2d_layer(inp, self.filters)
        for i in range(self.conv_layers):
            c = self.conv2d_layer(c, self.filters * (i+1))
        c = Conv2D(1, kernel_size=self.kernel_size, activation='relu')(c)

        c = Flatten()(c)
        # c = Dense(32, activation='relu')(c)
        c = Dense(2, activation='relu')(c)

        model = Model(inp, c)

        print(model.summary())

        # What should loss be?? think binary crossentropy is ok
        model.compile(loss='mse', optimizer=self.opt, metrics=['accuracy'])

        return model

    def train(self):
        start_time = datetime.datetime.now()
        for epoch in range(self.epochs):
            for batch_i, (imgs_A, target_vals) in enumerate(self.data_loader.load_batch_with_data(self.batch_size)):
                # train discriminator


                d_loss = self.cnn.train_on_batch(imgs_A, target_vals)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] time: %s" % (epoch,
                                                                                         self.epochs,
                                                                                         batch_i,
                                                                                         self.data_loader.n_batches,
                                                                                         d_loss[0],
                                                                                         100 * d_loss[1],
                                                                                         elapsed_time))

                if batch_i % self.sample_interval == 0:
                    plt.imshow(imgs_A[0])
                    d_output = self.cnn(imgs_A)[0]
                    print(d_output)
                    print(target_vals[0])
                    plt.plot(d_output[0], d_output[1], 'ro')
                    plt.plot(target_vals[0][0], target_vals[0][1], 'go')
                    plt.show()

                if epoch % self.model_save_interval == 0 and batch_i == 0:
                    self.cnn.save("models/model%d_%d" % (epoch, batch_i))


if __name__ == '__main__':
    cnn = ImageToLabel()
    cnn.train()
