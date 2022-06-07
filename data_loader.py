import itertools
import random
from glob import glob
from re import T
import numpy as np
# from skimage import transform
from PIL import Image
import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Concatenate


class DataLoader(Sequence):
    def __init__(self, config, data_to_use="pose_info/distances.npy", start=0, end=None):
        self.n_batches = 1
        self.data_folder = config.dataset_name
        self.img_res = (config.img_size, config.img_size)
        self.start_img = start
        if end==None:
            self.dataset_size = config.dataset_size #4500 #2338 #3
        else:
            self.dataset_size = end
        self.randomisations = config.randomisations #5
        self.data_to_use = data_to_use
        self.img_to_gen = "canonical"
        if self.data_to_use:
            self.data_file = np.load("%s/%s" % (self.data_folder, self.data_to_use))
            print(self.data_file)
        self.depth_file = "depth"
        self.segmap_file = "segmap"
        self.batch_size=config.batch_size

        self.use_rand = config.use_rand
        self.use_canon = config.use_canon
        self.use_depth = config.use_depth
        self.use_seg = config.use_seg

    def load_data(self, batch_size=1):
        img_nums = np.random.randint(self.dataset_size, size=batch_size)

        imgs_A = []
        imgs_B = []
        img_depths = []
        img_segmaps = []
        for i in range(len(img_nums)):
            randomisation = np.random.randint(self.randomisations)
            img_path_randomised = glob(
                '%s/randomised/%s/randomised_%s.png' % (self.data_folder, randomisation, img_nums[i]))

            while len(img_path_randomised) == 0:
                img_nums[i] = np.random.randint(self.dataset_size)
                img_path_randomised = glob('%s/randomised/%s/randomised_%s.png' % (
                    self.data_folder, randomisation, img_nums[i]))

            img_path_target = glob(
                '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[i]))

            print("IMG PATH: ", img_path_randomised)

            img_randomised = self.imread(img_path_randomised[0])
            img_target = self.imread(img_path_target[0])

            img_target = np.array(Image.fromarray(img_target).resize(self.img_res))
            img_randomised = np.array(Image.fromarray(img_randomised).resize(self.img_res))

            depth = np.load(glob("%s/%s/%s_%s.npy" % (self.data_folder, self.depth_file, self.depth_file, img_nums[i]))[0])
            segmap = np.load(glob("%s/%s/%s_%s.npy" % (self.data_folder, self.segmap_file, self.segmap_file, img_nums[i]))[0])

            depth = np.array(Image.fromarray(depth).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 1)
            segmap = np.array(Image.fromarray(segmap).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 1)

            imgs_A.append(img_target)
            imgs_B.append(img_randomised)
            img_depths.append(depth)
            img_segmaps.append(segmap)

        imgs_A = np.array(imgs_A)/ 127.5 - 1.
        imgs_B = np.array(imgs_B)/ 127.5 - 1.

        img_depths = np.array(img_depths)/np.max(img_depths)
        img_segmaps = np.array(img_segmaps)/np.max(img_segmaps)

        return imgs_A, imgs_B, img_depths, img_segmaps

    def load_batch(self, batch_size=1, is_testing=False):
        self.n_batches = int((self.dataset_size * self.randomisations) / batch_size)

        rand_order = list(itertools.product(range(self.dataset_size), range(self.randomisations)))
        np.random.shuffle(rand_order)

        for i in range(self.n_batches - 1):
            img_nums = rand_order[i * batch_size:i * batch_size + batch_size]

            imgs_target = []
            imgs_randomised = []
            img_depths = []
            img_segmaps = []

            for j in range(len(img_nums)):

                
                img_path_target = glob(
                    '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j][0]))

                while len(img_path_target) == 0:
                    img_nums[j] = (np.random.randint(self.dataset_size), img_nums[j][1])
                    img_path_target = glob(
                    '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j][0]))

                img_path_randomised = glob('%s/randomised/%s/randomised_%s.png' % (
                    self.data_folder, img_nums[j][1], img_nums[j][0]))
                img_randomised = self.imread(img_path_randomised[0])
                img_target = self.imread(img_path_target[0])


                img_target = np.array(Image.fromarray(img_target).resize(self.img_res))

                img_randomised = np.array(Image.fromarray(img_randomised).resize(self.img_res))

                depth = np.load(glob("%s/%s/%s_%s.npy" % (self.data_folder, self.depth_file, self.depth_file, img_nums[j][0]))[0])
                segmap = np.load(glob("%s/%s/%s_%s.npy" % (self.data_folder, self.segmap_file, self.segmap_file, img_nums[j][0]))[0])

                depth = np.array(Image.fromarray(depth).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 1)
                segmap = np.array(Image.fromarray(segmap).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 1)

                imgs_target.append(img_target)

                imgs_randomised.append(img_randomised)

                img_depths.append(depth)
                img_segmaps.append(segmap)
                

            imgs_target = np.array(imgs_target)/127.5 - 1.
            imgs_randomised = np.array(imgs_randomised)/ 127.5 - 1.

            img_depths = np.array(img_depths)/np.max(img_depths)# * 2 -1
            img_segmaps = np.array(img_segmaps)/np.max(img_segmaps)

            print("taget")

            yield imgs_target, imgs_randomised, img_depths, img_segmaps

    def imread(self, path):
        return np.array(Image.open(path).convert('RGB'))

    def __getitem__(self, i):
        self.n_batches = int(self.dataset_size / self.batch_size)

        for _ in range(self.n_batches - 1):
            img_nums = np.random.randint(low=self.start_img, high=self.dataset_size, size=self.batch_size)

            imgs_A = []
            target_coords = []
            for j in range(len(img_nums)):

                # get 
                img_path_target = glob(
                    '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j]))
                data_val = self.data_file[img_nums[j]]
                
                while (len(img_path_target)) == 0 or any(list(map(lambda x: x > 3 or (-0.1 < x < 0.1) or x < -3, data_val))):
                    # print(data_val, " is incorrect")
                    img_nums[j] = np.random.randint(self.dataset_size)
                    img_path_target = glob(
                        '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j]))
                
                    data_val = self.data_file[img_nums[j]]

                inputs_to_use = []

                if self.use_rand:
                    img_path_randomised = glob('%s/randomised/%s/randomised_%s.png' % (
                    self.data_folder, np.random.randint(0, self.randomisations), img_nums[j]))
                    img_randomised = self.imread(img_path_randomised[0])
                    img_randomised = np.array(Image.fromarray(img_randomised).resize(self.img_res))
                    img_randomised = np.array(img_randomised)/ 127.5 - 1.
                    inputs_to_use.append(img_randomised)
                if self.use_canon:
                    img_target = self.imread(img_path_target[0])
                    img_target = np.array(Image.fromarray(img_target).resize(self.img_res))
                    img_target = np.array(img_target)/ 127.5 - 1.
                    inputs_to_use.append(img_target)
                if self.use_seg:
                    segmap_file = np.load(glob("%s/%s/%s_%s.npy" % (self.data_folder, self.segmap_file, self.segmap_file, img_nums[j]))[0])
                    segmap = np.array(Image.fromarray(segmap_file).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 1)
                    segmap = np.array(segmap)/np.max(segmap)
                    inputs_to_use.append(segmap)
                if self.use_depth:
                    depth_file = np.load(glob("%s/%s/%s_%s.npy" % (self.data_folder, self.depth_file, self.depth_file, img_nums[j]))[0])
                    depth = np.array(Image.fromarray(depth_file).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 1)
                    depth = np.array(depth)/np.max(depth)
                    inputs_to_use.append(depth)

                combined_imgs = Concatenate(axis=-1)(inputs_to_use)

                imgs_A.append(combined_imgs)
                target_coords.append(data_val)

            # print("IMGS A")
            # print(np.array(imgs_A).shape)
            # print("target coords")
            # print(np.array(target_coords).shape)

            return np.array(imgs_A), np.array(target_coords)
            # return np.array([0,0,0])

    def __len__(self):
        return int(self.dataset_size / self.batch_size)

    def on_epoch_end(self):
        pass