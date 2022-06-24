import blenderproc as bproc
from tensorflow import keras
from scene_generator import Scene #, CanonicalParams, RandomisationParams
import numpy as np
from PIL import Image
from glob import glob

class SceneGenCallback(keras.callbacks.Callback):
    def __init__(self, sample_interval, c_params, r_params):
        super(SceneGenCallback, self).__init__()
        self.epoch = 0
        self.sample_interval = sample_interval
        self.canonical_params = c_params
        self.randomisation_params = r_params
        self.output_folder = "/vol/bitbucket/efb4518/fyp/fyp/img_to_label_test"
        self.img_res = (self.canonical_params.img_width, self.canonical_params.img_height)

        self.scene = Scene(self.canonical_params, self.randomisation_params, self.output_folder)

        self.blue_mat = None
        self.orange_mat = None
        self.distance = [0,0,0]

        bproc.camera.set_intrinsics_from_blender_params(image_height=self.scene.c_params.img_height,
                                                        image_width=self.scene.c_params.img_width)

        # bproc.renderer.enable_depth_output(activate_antialiasing=True)
        self.cube, self.poi, self.ceiling, self.box, blue_mat, orange_mat = self.scene.setup_scene(self.blue_mat, self.orange_mat)

        bproc.object.delete_multiple([self.box])

        self.blue_mat = blue_mat
        self.orange_mat = orange_mat
        cube_axes = bproc.object.create_empty("cube_axes", empty_type="plain_axes")
        cube_axes.set_local2world_mat(self.cube.get_local2world_mat())

        bproc.lighting.light_surface([self.ceiling], emission_strength=self.scene.c_params.ambient_light)

    def imread(self, path):
        return np.array(Image.open(path).convert('RGB'))

    def generate_example_scene(self, batch):
        
        cam_pose, visible, self.cam_loc, self.cam_rot = self.scene.generate_pos_bb_visible(self.poi, self.scene.r_params.min_camera_distance,
                                                             self.scene.r_params.max_camera_distance, 90,
                                                             bb=self.cube.get_bound_box(), obj=self.cube)
       
        bproc.camera.add_camera_pose(cam_pose)
        bproc.renderer.set_max_amount_of_samples(30)
        data = bproc.renderer.render()

        self.distance, rotation_matrix, euler_angles = self.scene.generate_data(end=self.poi, start=self.cam_loc, start_orient=np.eye(3), end_orient=self.cam_rot)

        self.scene.output_file(data, "randomised", self.epoch * 1000 + batch, 0)
        img_path_target = glob("%s/%s/%d/%s_%d.png" % (self.output_folder, "randomised", 0, "randomised",  self.epoch * 1000 + batch))
        img_target = self.imread(img_path_target[0])
        img_target = np.expand_dims(np.array(Image.fromarray(img_target).resize(self.img_res)), 0)

        print(img_target.shape)

        output = self.model(img_target, training=False)

        print(output)
        print(output.shape)

        axesz2 = bproc.object.create_primitive("CUBE")
        axesz2.set_scale([0.01,0.01,10])
        axesz2.set_location(self.cam_loc+self.distance)
        axesy2 = bproc.object.create_primitive("CUBE")
        axesy2.set_scale([0.01,10,0.01])
        axesy2.set_location(self.cam_loc+self.distance)
        axesx2 = bproc.object.create_primitive("CUBE")
        axesx2.set_scale([10,0.01,0.01])
        axesx2.set_location(self.cam_loc+self.distance)

        axesz = bproc.object.create_primitive("CUBE")
        axesz.set_scale([0.01,0.01,10])
        axesz.set_location(self.cam_loc + output[0])
        axesz.add_material(self.blue_mat)
        axesy = bproc.object.create_primitive("CUBE")
        axesy.set_scale([0.01,10,0.01])
        axesy.add_material(self.blue_mat)
        axesy.set_location(self.cam_loc + output[0])
        axesx = bproc.object.create_primitive("CUBE")
        axesx.set_scale([10,0.01,0.01])
        axesx.add_material(self.blue_mat)
        axesx.set_location(self.cam_loc + output[0])

        # axes2 = bproc.object.create_empty("plain_axes")
        # axes2.set_scale([10,10,10])
        # axes2.set_location(cam_loc - distance)

        # axesz2 = bproc.object.create_primitive("CUBE")
        # axesz2.set_scale([0.01,0.01,10])
        # axesz2.set_location([0,0,0])
        # axesy2 = bproc.object.create_primitive("CUBE")
        # axesy2.set_scale([0.01,10,0.01])
        # axesy2.set_location([0,0,0])
        # axesz2 = bproc.object.create_primitive("CUBE")
        # axesz2.set_scale([10,0.01,0.01])
        # axesz2.set_location([0,0,0])

        

        # bproc.utility.reset_keyframes()

        # bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc*1.5, cam_rot))

        data = bproc.renderer.render()
        self.scene.output_file(data, "prediction",  self.epoch * 1000 + batch, 0)

        bproc.utility.reset_keyframes()
        bproc.object.delete_multiple([axesx, axesy, axesz, axesx2, axesy2, axesz2])

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.sample_interval == 0:
            if batch == 0:
                self.epoch += 1
            self.generate_example_scene(batch)