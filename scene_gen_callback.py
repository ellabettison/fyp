import blenderproc as bproc
from tensorflow import keras
from scene_generator import Scene, CanonicalParams, RandomisationParams

class SceneGenCallback(keras.callbacks.Callback):
    def __init__(self, sample_interval):
        super(SceneGenCallback, self).__init__()
        self.sample_interval = sample_interval
        canonical_params = CanonicalParams()
        randomisation_params = RandomisationParams()

        self.scene = Scene(canonical_params, randomisation_params, "/vol/bitbucket/efb4518/fyp/fyp/img_to_label_test")

    def generate_example_scene(self, batch):
        bproc.camera.set_intrinsics_from_blender_params(image_height=self.scene.c_params.img_height,
                                                        image_width=self.scene.c_params.img_width)

        bproc.renderer.enable_depth_output(activate_antialiasing=True)
        cube, poi, ceiling, box, blue_mat = self.scene.setup_scene()
        cube_axes = bproc.object.create_empty("cube_axes", empty_type="plain_axes")
        cube_axes.set_local2world_mat(cube.get_local2world_mat())

        bproc.lighting.light_surface([ceiling], emission_strength=self.scene.c_params.ambient_light)

        cam_pose, visible, cam_loc, cam_rot = self.scene.generate_pos_bb_visible(poi, self.scene.r_params.min_camera_distance,
                                                             self.scene.r_params.max_camera_distance, 90,
                                                             bb=cube.get_bound_box(), obj=cube)
        bproc.camera.add_camera_pose(cam_pose)
        bproc.renderer.set_max_amount_of_samples(30)
        data = bproc.renderer.render()
        self.scene.output_file(data, "randomised", batch, 0)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.sample_interval == 0:
            self.generate_example_scene(batch)