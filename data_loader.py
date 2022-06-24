import itertools
import random
from glob import glob
from re import T
import numpy as np
# from skimage import transform
from PIL import Image
import tensorflow as tf
import cv2 as cv
from matplotlib import pyplot as plt
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
            self.dataset_size = config.dataset_size 
        else:
            self.dataset_size = end
        self.randomisations = config.randomisations
        self.data_to_use = data_to_use
        self.img_to_gen = "canonical"
        if self.data_to_use:
            self.distances_file = np.load("%s/%s" % (self.data_folder, self.data_to_use))
            print(self.distances_file)
            self.angles_file = np.load("%s/%s" % (self.data_folder, "pose_info/euler_angles.npy"))
        
        self.depth_file = "depth"
        self.segmap_file = "segmap"
        self.batch_size=config.batch_size

        self.use_rand = config.use_rand
        self.use_canon = config.use_canon
        self.use_depth = config.use_depth
        self.use_seg = config.use_seg
        self.use_keypoint = config.use_keypoint
        self.use_2d_keypoints = config.use_2d_keypoints
        self.use_vector = config.use_vector
        self.use_task_loss = config.use_task_loss

        if self.use_vector:
            self.vectors_file = np.load("%s/%s" % (self.data_folder, "pose_info/vector_reprs_no_skips.npy"))


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

    def load_batch(self, batch_size=1):
        self.n_batches = int((self.dataset_size * self.randomisations) / batch_size)

        rand_order = list(itertools.product(range(self.dataset_size), range(self.randomisations)))
        np.random.shuffle(rand_order)

        for i in range(self.n_batches - 1):
            img_nums = rand_order[i * batch_size:i * batch_size + batch_size]

            imgs_target = []
            imgs_randomised = []
            img_depths = []
            img_segmaps = []
            imgs_keypoints = []
            distances = []

            for j in range(len(img_nums)):

                
                img_path_target = glob(
                    '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j][0]))

                distances_val = []
                if self.use_task_loss:
                    distances_val = self.distances_file[img_nums[j][0]]

                while len(img_path_target) == 0 or (self.use_task_loss and any(list(map(lambda x: x > 3 or (-0.1 < x < 0.1) or x < -3, distances_val)))):
                    img_nums[j] = (np.random.randint(self.dataset_size), img_nums[j][1])
                    img_path_target = glob(
                    '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j][0]))
                    distances_val = self.distances_file[img_nums[j][0]]

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

                if self.use_keypoint:
                    img_keypoints= img_target.copy()
                    gray = cv.cvtColor(img_keypoints, cv.COLOR_RGB2GRAY)
                    gray = np.float32(gray)
                    corners = cv.cornerHarris(gray, 2, 3, 0.01)
                    dst = cv.dilate(corners, None)
                    # Threshold for an optimal value, it may vary depending on the image.
                    img_keypoints[dst > 0.02 * dst.max()] = [0, 0, 255]
                    img_keypoints = np.array(Image.fromarray(img_keypoints).resize(self.img_res))
                    imgs_keypoints.append(img_keypoints)

                if self.use_task_loss:
                    distances.append(distances_val)

                imgs_target.append(img_target)

                imgs_randomised.append(img_randomised)

                img_depths.append(depth)
                img_segmaps.append(segmap)
                

            imgs_target = np.array(imgs_target)/127.5 - 1.
            imgs_randomised = np.array(imgs_randomised)/ 127.5 - 1.
            imgs_keypoints = np.array(imgs_keypoints)/ 127.5 - 1.

            img_depths = np.array(img_depths)/np.max(img_depths)# * 2 -1
            img_segmaps = np.array(img_segmaps)/np.max(img_segmaps)
            distances = np.array(distances)

            yield imgs_target, imgs_randomised, img_depths, img_segmaps, imgs_keypoints, distances

    def imread(self, path):
        return np.array(Image.open(path).convert('RGB'))

    def __getitem__(self, i):
        self.n_batches = int(self.dataset_size / self.batch_size)
        img_path_no_dist_keypoints = []

        for _ in range(self.n_batches - 1):
            img_nums = np.random.randint(low=self.start_img, high=self.dataset_size, size=self.batch_size)

            vector_reprs = []
            imgs_A = []
            target_coords = []
            for j in range(len(img_nums)):

                img_path_target = glob(
                    '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j]))
                distances_val = self.distances_file[img_nums[j]]
                if self.use_keypoint:
                    img_path_no_dist_keypoints = glob(
                    '%s/%s/%s_%s.png' % ("generated_imgs_distractors", self.img_to_gen, self.img_to_gen, img_nums[j]))
                
                while (len(img_path_target)) == 0 or any(list(map(lambda x: x > 3 or (-0.1 < x < 0.1) or x < -3, distances_val))) or (self.use_keypoint and len(img_path_no_dist_keypoints) ==0):
                    img_nums[j] = np.random.randint(self.dataset_size)
                    img_path_target = glob(
                        '%s/%s/%s_%s.png' % (self.data_folder, self.img_to_gen, self.img_to_gen, img_nums[j]))
                
                    distances_val = self.distances_file[img_nums[j]]

                    if self.use_keypoint:
                        img_path_no_dist_keypoints = glob(
                        '%s/%s/%s_%s.png' % ("generated_imgs_distractors", self.img_to_gen, self.img_to_gen, img_nums[j]))

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
                if self.use_keypoint:
                    img_path_no_dist_keypoints = glob(
                    '%s/%s/%s_%s.png' % ("generated_imgs_distractors", self.img_to_gen, self.img_to_gen, img_nums[j]))
                    img_no_dist_keypoints = self.imread(img_path_no_dist_keypoints[0])
                    img_keypoints = self.imread(img_path_target[0])
                    gray = cv.cvtColor(img_no_dist_keypoints, cv.COLOR_RGB2GRAY)
                    gray = np.float32(gray)
                    corners = cv.cornerHarris(gray, 2, 3, 0.01)
                    dst = cv.dilate(corners, None)
                    # Threshold for an optimal value, it may vary depending on the image.
                    img_keypoints[dst > 0.02 * dst.max()] = [0, 0, 255]
                    img_keypoints = np.array(Image.fromarray(img_keypoints).resize(self.img_res)).reshape(self.img_res[0], self.img_res[1], 3)
                    img_keypoints = np.array(img_keypoints) / 127.5 - 1.
                    inputs_to_use.append(img_keypoints)
                if self.use_2d_keypoints:
                    img_2d_keypoints = self.imread(img_path_target[0])
                    gray = cv.cvtColor(img_2d_keypoints, cv.COLOR_RGB2GRAY)
                    gray = np.float32(gray)
                    corners = cv.cornerHarris(gray, 2, 3, 0.01)
                    dst = cv.dilate(corners, None)
                    keypoints_2d = np.zeros((img_2d_keypoints.shape[0], img_2d_keypoints.shape[1]))
                    keypoints_2d[dst > 0.1 * dst.max()] = 1
                    img_2d_keypoints = np.array(Image.fromarray(keypoints_2d).resize(self.img_res)).reshape(
                        (self.img_res[0],
                         self.img_res[1], 1))
                    inputs_to_use.append(img_2d_keypoints)
                if self.use_vector:
                    vector_reprs.append(self.vectors_file[img_nums[j]])

                
                if len(inputs_to_use) > 0:
                    combined_imgs = Concatenate(axis=-1)(inputs_to_use)
                else:
                    combined_imgs = []

                imgs_A.append(combined_imgs)
                target_coords.append(distances_val)

            inputs = {}
            inputs['img'] = np.array(imgs_A)
            inputs['vector'] = np.array(vector_reprs)

            return inputs, np.array(target_coords)

    def __len__(self):
        return int(self.dataset_size / self.batch_size)

    def on_epoch_end(self):
        pass