import blenderproc as bproc
import numpy as np
import random
import glob
from pathlib import Path
# import bpy
from PIL import Image

bproc.init()


class CanonicalParams:
    def __init__(self):
        self.ambient_light = 5  # 1000
        self.camera_distance = 19
        self.img_width = 256
        self.img_height = 256


class RandomisationParams:
    def __init__(self):
        self.min_ambient_light = 0.3
        self.max_ambient_light = 3
        self.min_spot_light = 0
        self.max_spot_light = 2000
        self.min_spot_distance = 3
        self.max_spot_distance = 19


class Scene:
    def __init__(self, canonical_params, randomisation_params):
        self.c_params = canonical_params
        self.r_params = randomisation_params

    def load_obj(self, obj_name, pos, rot, from_file=False):
        if from_file:
            obj = bproc.loader.load_obj(obj_name)[0]
        else:
            obj = bproc.object.create_primitive(obj_name)
        obj.set_location(pos)
        obj.set_rotation_euler(rot)

        return obj, bproc.object.compute_poi([obj])

    def set_spot_light(self, loc, energy):
        light = bproc.types.Light()
        light.set_location(loc)
        light.set_energy(energy)

    def set_camera(self, loc, poi):
        # Set the camera to be in front of the object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - loc, inplane_rot=0)
        cam_pose = bproc.math.build_transformation_mat(loc, rotation_matrix)
        bproc.camera.add_camera_pose(cam_pose)
        return cam_pose

    def output_file(self, data, folder):
        # Write the rendering into an hdf5 file
        bproc.writer.write_hdf5("output/%s/" % folder, data)
        arr_colours = np.array(data.get("colors")[0]).reshape((self.c_params.img_width, self.c_params.img_height, 3))
        im = Image.fromarray(arr_colours)
        im.save("output/%s/colours.png" % folder)

        if data.__contains__("depth"):
            arr_depth = np.array(data.get("depth")).reshape(self.c_params.img_width, self.c_params.img_height)
            np.save("output/depths/depth.npy", arr_depth)

        if data.__contains__("instance_segmaps"):
            arr_seg = np.array(data.get("instance_segmaps")).reshape(self.c_params.img_width, self.c_params.img_height)
            np.save("output/segmaps/segmap.npy", arr_seg)

    def generate_random_pos(self, height, min, max):
        return [np.random.randint(min, max), height, np.random.randint(min, max)]

    def generate_pos_distance_from_point(self, distance, radmax):
        az_angle = random.random() * radmax
        zen_angle = random.random() * (np.pi / 2)

        y_pos = np.cos(zen_angle) * distance
        o = np.sin(zen_angle) * distance
        x_pos = np.cos(az_angle) * o
        z_pos = np.sin(az_angle) * o

        return [x_pos, y_pos, z_pos+1]

    def randomise_material(self, obj):
        # Find all materials
        materials = bproc.material.collect_all()
        # Collect all jpg images in the specified directory
        images = glob.glob("textures/textures/obj_textures/*.png")
        for mat in materials:
            # Load one random image
            random_mat = random.choice(images)
            print("random mat: ", random_mat)
            texture = bproc.loader.load_texture(random_mat)
            mat.infuse_texture(texture[0], mode='set', texture_scale=0.5)

    def render_scene(self):
        bproc.utility.reset_keyframes()

        # load canonical scene
        cube, poi = self.load_obj("CUBE", [0, 0, 1], [0, 0, 0])
        cube.add_uv_mapping("cube", overwrite=True)
        box, _ = self.load_obj("blender_objects/box_tall2.obj", [0, 0, 0], [np.pi/2, 0, 0], True)
        box.add_uv_mapping("cube", overwrite=True)
        print("box bb: ", box.get_bound_box())
        box.set_scale((2, 2, 2))
        ceiling, _ = self.load_obj("PLANE", [0, 0, 40], [0, 0, 0])
        ceiling.set_scale((20,20,1))
        print("plane bb: ", ceiling.get_bound_box())
        bproc.lighting.light_surface([ceiling], emission_strength=self.c_params.ambient_light)
        camera_loc = self.generate_pos_distance_from_point(self.c_params.camera_distance, np.pi / 2)
        cam_pose = self.set_camera(camera_loc, poi)

        # Render the scene
        bproc.camera.set_intrinsics_from_blender_params(image_height=self.c_params.img_height,
                                                        image_width=self.c_params.img_width)
        bproc.renderer.set_max_amount_of_samples(30)
        data = bproc.renderer.render()
        self.output_file(data, "canonical")
        bproc.utility.reset_keyframes()

        # randomise scene
        bproc.lighting.light_surface([ceiling], emission_strength=np.random.uniform(self.r_params.min_ambient_light,
                                                                                    self.r_params.max_ambient_light))
        self.randomise_material(box)
        self.set_spot_light(self.generate_pos_distance_from_point(np.random.uniform(self.r_params.min_spot_distance,
                                                                                    self.r_params.max_spot_distance),
                                                                  np.pi / 2),
                            np.random.uniform(self.r_params.min_spot_light, self.r_params.max_spot_light))

        # Render the scene
        bproc.camera.add_camera_pose(cam_pose)
        # bproc.renderer.enable_depth_output(activate_antialiasing=False)
        data = bproc.renderer.render()
        # data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))
        self.output_file(data, "randomised")


canonical_params = CanonicalParams()
randomisation_params = RandomisationParams()

scene = Scene(canonical_params, randomisation_params)
scene.render_scene()
