import blenderproc as bproc
import numpy as np
import random
import glob
from pathlib import Path
# import bpy

bproc.init()


class Scene:
    def load_obj(self, obj_name, pos, rot, from_file=False, randomise_material=False):
        if from_file:
            obj = bproc.loader.load_obj(obj_name)[0]
            print("objects: ", obj)
        else:
            obj = bproc.object.create_primitive(obj_name)
        obj.set_location(pos)
        obj.set_rotation_euler(rot)

        # self.materials = bproc.material.collect_all()
        # print(self.materials)

        # obj.set_material(0, random.choice(self.materials))

        return obj, bproc.object.compute_poi([obj])

    def set_spot_light(self, loc, energy):
        light = bproc.types.Light()
        light.set_location(loc)
        light.set_energy(energy)

    def set_camera(self, loc, poi):
        # Set the camera to be in front of the object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - loc, inplane_rot=-2.5)
        cam_pose = bproc.math.build_transformation_mat(loc, rotation_matrix)
        bproc.camera.add_camera_pose(cam_pose)

    def output_file(self, data):
        # Write the rendering into an hdf5 file
        bproc.writer.write_hdf5("output/", data)
        # bproc.writer.write_bop("output/imgs/",data)

    def generate_random_pos(self, height, min, max):
        return [np.random.randint(min, max), height, np.random.randint(min, max)]

    def generate_pos_distance_from_point(self, distance, radmax):
        az_angle = random.random() * radmax
        zen_angle = random.random() * (np.pi / 2)

        y_pos = np.cos(zen_angle) * distance
        o = np.sin(zen_angle) * distance
        x_pos = np.cos(az_angle) * o
        z_pos = np.sin(az_angle) * o

        print(x_pos, y_pos, z_pos)

        return x_pos, y_pos, z_pos

    def randomise_material(self, obj):
        # Find all materials
        materials = bproc.material.collect_all()
        # Collect all jpg images in the specified directory
        images = glob.glob("textures/textures/obj_textures/*.png")
        # print("Images: ", images)
        # for mat in materials:
        # Load one random image
        #     image = bpy.data.images.load(filepath=str(random.choice(images)))
        # Set it as base color of the current material
        #     mat.set_principled_shader_value("Base Color", image)

    def set_ambient_light(self, brightness):
        ceiling, _ = self.load_obj("blender_objects/plane.obj", [0, 30, 0], [0, 0, 0], True, True)
        bproc.lighting.light_surface([ceiling], emission_strength=brightness)

    def render_scene(self):
        cube, poi = self.load_obj("CUBE", [0, 1, 0], [0, 0, 0])  # self.generate_random_pos(2, -3, 3), [0, 0, 0])
        _, _ = self.load_obj("SPHERE", [2, 1, 2], [0, 0, 0])

        box, _ = self.load_obj("blender_objects/box_tall.obj", [0, 0, 0], [0, 0, 0], True, True)
        box.set_scale((2, 2, 2))
        print(box.get_bound_box())

        self.randomise_material(box)

        self.set_spot_light([2, 10, 0], np.random.randint(200,700))
        self.set_ambient_light(np.random.random()*6.0)

        camera_loc = self.generate_pos_distance_from_point(15, np.pi / 2)

        self.set_camera(camera_loc, poi)
        # Render the scene
        bproc.renderer.set_max_amount_of_samples(30)
        data = bproc.renderer.render()

        self.output_file(data)


scene = Scene()
scene.render_scene()
