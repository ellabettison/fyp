

import blenderproc as bproc

import sys

sys.path.append('/vol/bitbucket/efb4518/fyp/fyp')

import argparse
from peg_predictions_gan import RCAN
from image_to_label import ImageToLabel
from data_loader import DataLoader

# functions_dict = {
#     "gan": run_gan,
#     "pose": run_image_to_label
# }

class Main:
    def get_args(self):
        parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-i", "--img_size", default=128, type=int, help="Shape of image to process")
        parser.add_argument("-f", "--filters", default=8, type=int, help="Filters to use for conv layers")
        parser.add_argument("-k", "--kernel_size", default=3, type=int, help="Kernel size to use for conv layers")
        parser.add_argument("-b", "--batch_size", default=10, type=int, help="Batch size")
        parser.add_argument("-e", "--epochs", default=20, type=int, help="Number of epochs")
        parser.add_argument("-d", "--dataset_name", default="/vol/bitbucket/efb4518/fyp/fyp/generated_imgs_numpy", type=str, help="Folder from which to extract canonical, randomised images etc")
        parser.add_argument("-o", "--output_folder", default="/vol/bitbucket/efb4518/fyp/fyp/generated_samples2", type=str, help="Folder in which to store examples of generated data")
        parser.add_argument("-l", "--learning_rate", default=0.0002, type=float, help="Learning rate")
        parser.add_argument("-p", "--sample_interval", default=10, type=int)
        parser.add_argument("-n", "--dataset_size", default=4500, type=int, help="How much of dataset to use for training")
        parser.add_argument("-r", "--randomisations", default=5, type=int, help="Number of randomisations to use for training")
        parser.add_argument("-w", "--loss_weights", default=[1,1,1,0,0], nargs="+", type=int)
        parser.add_argument("-s", "--use_seg", action="store_true")
        parser.add_argument("-q", "--use_depth", action="store_true")
        parser.add_argument("-v", "--use_vector", action="store_true")
        parser.add_argument("-c", "--use_canon", action="store_true")
        parser.add_argument("-u", "--use_rand", action="store_true")
        parser.add_argument("-x", "--repetitions", default=10, type=int, help="Number of repetitions to average")
        parser.add_argument("-g", "--graph_name", default="train_curve", type=str)

        args = parser.parse_args()
        # config = vars(args)
        return args

    def run_gan(self, config):
        rcan = RCAN(config, DataLoader(config))
        rcan.train()

    def run_img_to_label(self, config):
        img_to_label = ImageToLabel(config)
        img_to_label.train()

if __name__ == "__main__":
    main_prog = Main()
    config = main_prog.get_args()

    print("runnnign")

    cnn = ImageToLabel(config)
    cnn.train()
