import random
from glob import glob
import numpy as np
from skimage import transform
from PIL import Image
import tensorflow as tf


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128), data_to_use=None):
        self.dataset_name = dataset_name
        self.img_res = img_res
        # self.dataset_size = 29961
        self.dataset_size = 1000
        self.randomisations = 5
        self.data_to_use = data_to_use
        if self.data_to_use:
            self.data_file = np.load("datasets/%s/reference_env/0/%s" % (self.dataset_name, self.data_to_use))
            print(np.max(self.data_file[:,1]))
        # Convert to Lab colourspace
        # srgb_p = ImageCms.createProfile("sRGB")
        # lab_p = ImageCms.createProfile("LAB")
        #
        # self.rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        # self.lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")

    def load_data(self, batch_size=1):
        img_nums = np.random.randint(self.dataset_size, size=batch_size)

        # path = glob('./datasets/%s/%s/*' % (self.dataset_name))
        # batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_num in img_nums:
            randomisation = np.random.randint(self.randomisations)
            img_path_randomised = glob(
                'datasets/%s/different_randomisations/%s/RGB/11%s.png' % (self.dataset_name, randomisation, img_num))
            img_path_target = glob(
                'datasets/%s/reference_env/0/RGB/11%s.png' % (self.dataset_name, img_num))
            img_randomised = self.imread(img_path_randomised[0])
            img_target = self.imread(img_path_target[0])

            # h, w, _ = img.shape
            # _w = int(w / 2)
            # img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_target = np.array(Image.fromarray(img_target).resize(self.img_res))
            # img_A = scipy.misc.imresize(img_A, self.img_res)
            # img_B = scipy.misc.imresize(img_B, self.img_res)
            img_randomised = np.array(Image.fromarray(img_randomised).resize(self.img_res))

            # If training => do random flip
            # if not is_testing and np.random.random() < 0.5:
            #     img_A = np.fliplr(img_A)
            #     img_B = np.fliplr(img_B)

            imgs_A.append(img_target)
            imgs_B.append(img_randomised)

        imgs_A = np.array(imgs_A) / 127.5 - 1.
        imgs_B = np.array(imgs_B) / 127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        # data_type = "train" if not is_testing else "val"
        # path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        # random.shuffle(path)

        self.n_batches = int(self.dataset_size / batch_size)

        for i in range(self.n_batches - 1):
            img_nums = np.random.randint(self.dataset_size, size=batch_size)

            # path = glob('./datasets/%s/%s/*' % (self.dataset_name))
            # batch_images = np.random.choice(path, size=batch_size)

            imgs_A = []
            imgs_B = []
            for img_num in img_nums:
                randomisation = np.random.randint(self.randomisations)
                img_path_randomised = glob('datasets/%s/different_randomisations/%s/RGB/11%s.png' % (
                    self.dataset_name, randomisation, img_num))
                # print(img_path_randomised)
                # print(img_num, randomisation, '/datasets/%s/different_randomisations/%s/RGB/11%s.png' % (
                #     self.dataset_name, randomisation, img_num))
                img_path_target = glob(
                    'datasets/%s/reference_env/0/RGB/11%s.png' % (self.dataset_name, img_num))
                img_randomised = self.imread(img_path_randomised[0])
                img_target = self.imread(img_path_target[0])

                # h, w, _ = img.shape
                # _w = int(w / 2)
                # img_A, img_B = img[:, :_w, :], img[:, _w:, :]

                img_target = np.array(Image.fromarray(img_target).resize(self.img_res))
                # img_A = scipy.misc.imresize(img_A, self.img_res)
                # img_B = scipy.misc.imresize(img_B, self.img_res)
                img_randomised = np.array(Image.fromarray(img_randomised).resize(self.img_res))

                # If training => do random flip
                # if not is_testing and np.random.random() < 0.5:
                #     img_A = np.fliplr(img_A)
                #     img_B = np.fliplr(img_B)

                imgs_A.append(img_target)
                imgs_B.append(img_randomised)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    def imread(self, path):
        return np.array(Image.open(path).convert('RGB'))
        # return np.array(color.rgb2lab(Image.open(path).convert('RGB')))
        # return imageio.imread(path, as_gray=False, pilmode="RGB").astype(np.float)
        # return color.rgb2lab(io.imread(path, pilmode='RGB'))

    def load_batch_with_data(self, batch_size):
        self.n_batches = int(self.dataset_size / batch_size)

        for i in range(self.n_batches - 1):
            img_nums = np.random.randint(self.dataset_size, size=batch_size)

            imgs_A = []
            target_coords = []
            for img_num in img_nums:
                img_path_target = glob(
                    'datasets/%s/reference_env/0/RGB/11%s.png' % (self.dataset_name, img_num))
                img_target = self.imread(img_path_target[0])

                img_target = np.array(Image.fromarray(img_target).resize((128,72)))

                imgs_A.append(img_target)
                target_coord = self.data_file[img_num+11000]
                target_coords.append([(target_coord[0]/848) * 128, (target_coord[1]/480)*72])
                # print(imgs_B[-1])

            imgs_A = np.array(imgs_A) / 127.5 - 1.

            yield imgs_A, tf.convert_to_tensor((target_coords), dtype=tf.float32)
