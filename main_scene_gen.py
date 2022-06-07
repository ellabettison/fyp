import blenderproc as bproc

import sys

sys.path.append('/vol/bitbucket/efb4518/fyp/fyp')

import argparse
from scene_generator import Scene

# functions_dict = {
#     "gan": run_gan,
#     "pose": run_image_to_label
# }

class CanonicalParams:
    def __init__(self):
        self.ambient_light = 5  # 1000
        self.camera_distance = 2
        self.img_width = 128
        self.img_height = 128


class RandomisationParams:
    def __init__(self):
        self.min_ambient_light = 0.5
        self.max_ambient_light = 5
        self.min_spot_light = 0
        self.max_spot_light = 200
        self.min_spot_distance = 1
        self.max_spot_distance = 3
        self.min_camera_distance = 1.2
        self.max_camera_distance = 2
        self.camera_perturbation_sd = 0.05  # 0.05
        self.no_randomisations = 5
        self.distractor_objects_radius = 2
        self.obj_size_range = 0.2
        self.min_obj_size = 0.05

class Main:
    def get_args(self):
        parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-i", "--img_size", default=128, type=int, help="Shape of image to process")
        parser.add_argument("-o", "--output_folder", default="/vol/bitbucket/efb4518/fyp/fyp/generated_imgs_distractors", type=str, help="Folder in which to store examples of generated data")
        parser.add_argument("-p", "--sample_interval", default=10, type=int)
        parser.add_argument("-n", "--dataset_size", default=4500, type=int, help="How much of dataset to use for training")
        parser.add_argument("-r", "--randomisations", default=5, type=int, help="Number of randomisations to generate")
        parser.add_argument("-s", "--use_seg", action="store_true")
        parser.add_argument("-q", "--use_depth", action="store_true")
        parser.add_argument("-v", "--use_vector", action="store_true")
        parser.add_argument("-c", "--use_canon", action="store_true")
        parser.add_argument("-m", "--no_images", default=10000, type=int, help="Number of images to generate")
        parser.add_argument("-g", "--rand_objs_to_gen", default=10, type=int, help="Number of random objects to add to scene")

        args = parser.parse_args()
        # config = vars(args)
        return args

    def gen_scenes(self, config):
        canonical_params = CanonicalParams()
        randomisation_params = RandomisationParams()

        randomisation_params.no_randomisations = config.randomisations
        canonical_params.img_width = config.img_size
        canonical_params.img_height = config.img_size

        scene = Scene(canonical_params, randomisation_params, config)
        scene.render_scene(no_images=config.no_images)


if __name__ == "__main__":
    main_prog = Main()
    config = main_prog.get_args()

    print("runnnign")

    main_prog.gen_scenes(config)
