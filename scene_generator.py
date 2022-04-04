
import blenderproc as bproc
import numpy as np
import glob
import os
from pathlib import Path
# import bpy
from PIL import Image
# from matplotlib import cm
import matplotlib.pyplot as plt

bproc.init()


class CanonicalParams:
    def __init__(self):
        self.ambient_light = 5  # 1000
        self.camera_distance = 2
        self.img_width = 64
        self.img_height = 64


class RandomisationParams:
    def __init__(self):
        self.min_ambient_light = 0.5
        self.max_ambient_light = 5
        self.min_spot_light = 0
        self.max_spot_light = 200
        self.min_spot_distance = 1
        self.max_spot_distance = 3
        self.min_camera_distance = 1
        self.max_camera_distance = 2
        self.camera_perturbation_sd = 0.05  # 0.05
        self.no_randomisations = 5


class Scene:
    def __init__(self, canonical_params, randomisation_params, output_folder):
        self.c_params = canonical_params
        self.r_params = randomisation_params
        self.images = glob.glob("textures/textures/obj_textures/*.png")
        self.orange_mat = glob.glob("blender_objects/orange_texture.png")[0]
        self.blue_mat = glob.glob("blender_objects/blue_texture.png")[0]
        self.output_folder = output_folder

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
        return light

    def set_camera(self, loc, poi, randomise=False):
        # Set the camera to be in front of the object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - loc, inplane_rot=0)

        # Add randomisation
        if randomise:
            randomisation = np.random.normal(0, self.r_params.camera_perturbation_sd, size=rotation_matrix.shape)
            rotation_matrix += randomisation
            # print("Randomisation: ", randomisation)
            # print(rotation_matrix)

        cam_pose = bproc.math.build_transformation_mat(loc, rotation_matrix)
        return cam_pose

    def output_file(self, data, folder, i, j):
        # Write the rendering into an hdf5 file
        # bproc.writer.write_hdf5("output/%s/" % folder, data)
        arr_colours = np.array(data.get("colors")[0]).reshape((self.c_params.img_width, self.c_params.img_height, 3))
        im = Image.fromarray(arr_colours)
        if folder == "randomised":
            im.save("%s/%s/%d/%s_%d.png" % (self.output_folder, folder, j, folder, i))
        else:
            im.save("%s/%s/%s_%d.png" % (self.output_folder, folder, folder, i))

        if data.__contains__("depth"):
            arr_depth = np.array(data.get("depth")).reshape((self.c_params.img_width, self.c_params.img_height))
            arr_depth /= np.max(arr_depth)

            # np.save("output/depths/depth_%d.npy" % i, arr_depth)
            print("DEPTH: ", arr_depth.shape)
            plt.imsave("%s/depths/depth_%d.png" % (self.output_folder, i), arr_depth, cmap="turbo")
            # im = Image.fromarray(np.uint8(cm._colormaps(arr_colours) * 255))
            # im.save("output/depths/depth_%d.png" % i, arr_depth)

        if data.__contains__("instance_segmaps"):
            arr_seg = np.array(data.get("instance_segmaps"), dtype=np.float64).reshape(
                (self.c_params.img_width, self.c_params.img_height))
            arr_seg /= np.max(arr_seg)
            plt.imsave("%s/segmaps/segmap_%d.png" % (self.output_folder, i), arr_seg, cmap="turbo")
            # np.save("output/segmaps/segmap_%d.npy" % i, arr_seg)

    # def generate_random_pos(self, height, min, max):
    #     return [np.random.randint(min, max), height, np.random.randint(min, max)]

    # def generate_pos_distance_from_point(self, distance, radmax):
    #     az_angle = random.random() * radmax
    #     zen_angle = random.random() * (np.pi / 2)
    #
    #     y_pos = np.cos(zen_angle) * distance
    #     o = np.sin(zen_angle) * distance
    #     x_pos = np.cos(az_angle) * o
    #     z_pos = np.sin(az_angle) * o
    #
    #     return [x_pos, y_pos, z_pos+1]

    def generate_pos_distance_from_point(self, center, radmin, radmax, azmax_angle):
        elevation_min = 10
        azimuth_min = -180
        azimuth_max = azimuth_min + azmax_angle
        return bproc.sampler.shell(center, radius_min=radmin, radius_max=radmax, elevation_min=elevation_min,
                                   azimuth_min=azimuth_min, azimuth_max=azimuth_max)

    def generate_pos_bb_visible(self, center, radmin, radmax, azmax_angle, bb, obj, max_samples=10, randomise=True):
        point = self.generate_pos_distance_from_point(center, radmin, radmax, azmax_angle)
        cam_pose = self.set_camera(point, center, randomise=randomise)
        i = 0
        while not self.check_if_bb_in_view(bb, cam_pose):
            point = self.generate_pos_distance_from_point(center, radmin, radmax, azmax_angle)
            cam_pose = self.set_camera(point, center, randomise=randomise)
            i += 1

            # return non-randomised point
            if i > max_samples:
                return self.set_camera(point, center, randomise=False), False
        return cam_pose, True

    def randomise_material(self, obj):
        # Find all materials
        # Collect all jpg images in the specified directory

        #
        # for mat in materials:
        #     # Load one random image
        new_mat = bproc.material.create("material")
        texture = None

        while texture is None or len(texture) == 0:
            random_mat = np.random.choice(self.images)
            texture = bproc.loader.load_texture(random_mat)

            if texture is None or len(texture) == 0:
                continue

            new_mat.infuse_texture(texture[0], mode='set', texture_scale=1)

            obj.clear_materials()
            obj.add_material(new_mat)

    def check_if_bb_in_view(self, bb, cam_pose):
        points = []
        for point in bb:
            p = bproc.object.create_primitive("SPHERE")
            p.set_scale((0.5, 0.5, 0.5))
            p.set_location(point)
            points.append(p)

        vis_objects = bproc.camera.visible_objects(cam_pose, sqrt_number_of_rays=100)
        # print("vis objs: ", vis_objects)
        objs_vis = vis_objects.intersection(points)
        print("len objs vis: ", len(objs_vis))
        bproc.object.delete_multiple(points)
        return len(objs_vis) >= 6

    # def generate_checkerboard(self):
    #     # a4_size = (2.97, 2.10)
    #     a4_size = (0.297, 0.210)
    #     board = bproc.object.create_primitive("PLANE")
    #     board.set_scale([a4_size[0], a4_size[1], 1])
    #     board.set_location([0, 0, 0.02])
    #     board.add_uv_mapping("cube", overwrite=True)
    #     mat = bproc.material.create("checkerboard")
    #     texture = bproc.loader.load_texture("Checkerboard-A4-25mm-10x7.png")
    #     mat.infuse_texture(texture[0], mode="set")  # , texture_scale=1)
    #     # board.add_material(mat)
    #     board.clear_materials()
    #     board.add_material(mat)
    #     return board

    # def add_apriltag(self, obj):
    #     mat = bproc.material.create("checkerboard")
    #     texture = bproc.loader.load_texture("apriltag.png")
    #     mat.infuse_texture(texture[0], mode="set", texture_scale=1)
    #     obj.clear_materials()
    #     obj.add_material(mat)

    # def print_camera_params_pose(self, obj=None):
    #     print("Camera K matrix: ", bproc.camera.get_intrinsics_as_K_matrix())
    #     # print("Object pose: ", obj.get_local2world_mat())

    # def generate_env_with_checkerboard(self):
    #     bproc.camera.set_intrinsics_from_blender_params(image_height=self.c_params.img_height,
    #                                                     image_width=self.c_params.img_width)
    #
    #     K = [[355, 0, 127.5],
    #          [0, 355, 127.5],
    #          [0, 0, 1]]
    #
    #     self.generate_environment()
    #     self.generate_checkerboard()
    #
    #     cube, poi = self.load_obj("CUBE", [0, 0, 0.1], [0, 0, 0])
    #
    #     random.seed(0)
    #     np.random.seed(0)
    #
    #     cam_pose = self.generate_pos_bb_visible(poi, self.r_params.min_camera_distance,
    #                                             self.r_params.max_camera_distance, 90,
    #                                             bb=cube.get_bound_box(), obj=cube, randomise=False)
    #
    #     bproc.camera.set_intrinsics_from_K_matrix(K, image_height=self.c_params.img_height,
    #                                               image_width=self.c_params.img_width)
    #
    #     # cam_pose = self.set_camera(camera_loc, poi)
    #     bproc.camera.add_camera_pose(cam_pose)
    #
    #     # print("IS IN VIEW!! ", self.check_if_bb_in_view(cube.get_bound_box(), cam_pose))
    #
    #     cube.delete()
    #
    #     # Render the scene
    #     bproc.renderer.set_max_amount_of_samples(30)
    #     data = bproc.renderer.render()
    #     self.output_file(data, "img_with_checkerboard", 9)
    #
    #     self.print_camera_params_pose()
    #     bproc.camera.get_camera_pose()
    #     print("Camera pose: ", bproc.camera.get_camera_pose())
    #     bproc.utility.reset_keyframes()

    def generate_environment(self):
        box, _ = self.load_obj("blender_objects/box_tall2.obj", [0, 0, 0], [np.pi / 2, 0, 0], True)
        box.add_uv_mapping("cube", overwrite=True)
        box.clear_materials()

        # box.set_scale((2, 2, 2))

        box.set_scale((0.2, 0.2, 0.2))
        print("box bb before: ", box.get_bound_box()[0][2])
        box.set_location([0, 0, 0])

        print("local to world mat ", box.get_origin())


        print("box bb: ", box.get_bound_box())
        ceiling, _ = self.load_obj("PLANE", [0, 0, 40], [0, 0, 0])
        ceiling.set_scale((20, 20, 1))
        print("plane bb: ", ceiling.get_bound_box())
        bproc.lighting.light_surface([ceiling], emission_strength=self.c_params.ambient_light)

        return ceiling, box

    def randomise_env(self, ceiling, box, poi):
        # randomise scene
        bproc.lighting.light_surface([ceiling], emission_strength=np.random.uniform(self.r_params.min_ambient_light,
                                                                                    self.r_params.max_ambient_light))

        self.randomise_material(box)
        light = self.set_spot_light(self.generate_pos_distance_from_point(poi, self.r_params.min_spot_distance,
                                                                          self.r_params.max_spot_distance, 360),
                                    np.random.uniform(self.r_params.min_spot_light, self.r_params.max_spot_light))
        return light

    def render_scene(self, no_images):
        bproc.camera.set_intrinsics_from_blender_params(image_height=self.c_params.img_height,
                                                        image_width=self.c_params.img_width)

        bproc.renderer.enable_depth_output(activate_antialiasing=True)
        # bproc.renderer.set_output_format("PNG")

        # load canonical scene
        cube, poi = self.load_obj("CUBE", [0, 0, 0.1], [0, 0, 0])
        cube.add_uv_mapping("cube", overwrite=True)
        cube.set_scale((0.1, 0.1, 0.1))
        cube.clear_materials()

        print("CUBE BB: ", cube.get_bound_box())

        ceiling, box = self.generate_environment()

        orange_mat = bproc.material.create("orange_mat")
        orange_texture = bproc.loader.load_texture(self.orange_mat)
        orange_mat.infuse_texture(orange_texture[0], mode='set', texture_scale=1)
        cube.add_material(orange_mat)

        blue_mat = bproc.material.create("blue_mat")
        blue_texture = bproc.loader.load_texture(self.blue_mat)
        blue_mat.infuse_texture(blue_texture[0], mode='set', texture_scale=1)

        for i in range(no_images):
            # 100= random number chosen to ensure no overlap when mutliple materials have to
            # be generated
            os.environ["BLENDER_PROC_RANDOM_SEED"] = str(i*self.r_params.no_randomisations*100)
            np.random.seed(i*self.r_params.no_randomisations*100)

            bproc.utility.reset_keyframes()

            box.clear_materials()
            box.add_material(blue_mat)
            bproc.lighting.light_surface([ceiling], emission_strength=self.c_params.ambient_light)

            cam_pose, visible = self.generate_pos_bb_visible(poi, self.r_params.min_camera_distance,
                                                             self.r_params.max_camera_distance, 90,
                                                             bb=cube.get_bound_box(), obj=cube)

            if not visible:
                continue

            bproc.camera.add_camera_pose(cam_pose)

            # Render the scene
            bproc.renderer.set_max_amount_of_samples(30)
            data = bproc.renderer.render()
            data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

            self.output_file(data, "canonical", i, 0)

            for j in range(randomisation_params.no_randomisations):

                bproc.utility.reset_keyframes()

                # randomise environment
                light = self.randomise_env(ceiling, box, poi)

                # Render the randomised scene
                bproc.camera.add_camera_pose(cam_pose)

                data = bproc.renderer.render()
                # bproc.postprocessing.remove_segmap_noise()
                self.output_file(data, "randomised", i, j)

                light.delete()


# apriltag
canonical_params = CanonicalParams()
randomisation_params = RandomisationParams()

scene = Scene(canonical_params, randomisation_params, "generated_imgs")
scene.render_scene(no_images=10000)
# scene.generate_env_with_checkerboard()

# TODO: use seeds for each scene - check if it works????
# TODO: store poses
