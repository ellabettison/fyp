
import blenderproc as bproc
from matplotlib import rc_params
import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import rand

bproc.init()



class Scene:
    def __init__(self, canonical_params, randomisation_params, config):
        self.c_params = canonical_params
        self.r_params = randomisation_params
        self.images = glob.glob("textures/obj_textures/*.png")
        self.orange_mat = glob.glob("blender_objects/orange_texture.png")[0]
        self.blue_mat = glob.glob("blender_objects/blue_texture.png")[0]
        self.green_mat = glob.glob("blender_objects/green_texture.png")[0]
        self.output_folder = config.output_folder
        self.start_img = config.start_img
        self.face_centres = []
        self.save_arrs_interval = config.sample_interval
        self.rand_objs_to_gen = config.rand_objs_to_gen

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

        cam_pose = bproc.math.build_transformation_mat(loc, rotation_matrix)
        return cam_pose, rotation_matrix


    def generate_data(self, end,start,start_orient, end_orient):
        distance=  end-start
        print(start_orient)
        rotation_matrix = np.matmul(np.linalg.inv(start_orient), end_orient)
        return distance,rotation_matrix, self.rot2eul(rotation_matrix)

    def output_file(self, data, folder, i, j):
        arr_colours = np.array(data.get("colors")[0]).reshape((self.c_params.img_width, self.c_params.img_height, 3))
        im = Image.fromarray(arr_colours)
        if folder == "randomised":
            im.save("%s/%s/%d/%s_%d.png" % (self.output_folder, folder, j, folder, i))
        else:
            im.save("%s/%s/%s_%d.png" % (self.output_folder, folder, folder, i))

        if data.__contains__("depth"):
            arr_depth = np.array(data.get("depth")).reshape((self.c_params.img_width, self.c_params.img_height))
            np.save("%s/depth/depth_%d.npy" % (self.output_folder, i), arr_depth)


        if data.__contains__("instance_segmaps"):
            arr_seg = np.array(data.get("instance_segmaps"), dtype=np.float64).reshape(
                (self.c_params.img_width, self.c_params.img_height))
            np.save("%s/segmap/segmap_%d.npy" % (self.output_folder, i), arr_seg)

    def generate_pos_distance_from_point(self, center, radmin, radmax, azmax_angle):
        elevation_min = 10
        azimuth_min = -180
        azimuth_max = azimuth_min + azmax_angle
        return bproc.sampler.shell(center, radius_min=radmin, radius_max=radmax, elevation_min=elevation_min,
                                   azimuth_min=azimuth_min, azimuth_max=azimuth_max)

    def generate_pos_bb_visible(self, center, radmin, radmax, azmax_angle, bb, obj, max_samples=40, randomise=True):
        point = self.generate_pos_distance_from_point(center, radmin, radmax, azmax_angle)
        cam_pose, rotation = self.set_camera(point, center, randomise=randomise)
        i = 0
        while not self.check_if_bb_in_view(obj, cam_pose):
            point = self.generate_pos_distance_from_point(center, radmin, radmax, azmax_angle)
            cam_pose, rotation = self.set_camera(point, center, randomise=randomise)
            i += 1

            # return non-randomised point
            if i > max_samples:
                cam_pose, rotation = self.set_camera(point, center, randomise=False)
                return cam_pose, False, point, rotation
        return cam_pose, True, point, rotation

    def randomise_material(self, obj):
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

    def get_face_centres(self, obj):
        face_centres = []
        centre = obj.get_location()
        size = abs(obj.get_bound_box()[0,0])
        for xyz in range(3):
            for negpos in [-1, 1]:
                face_centre = centre.copy()
                face_centre[xyz] += negpos * size
                face_centres.append(face_centre)
        return face_centres

    def check_if_bb_in_view(self, obj, cam_pose):
        points_list = []

        for face_no in range(len(self.face_centres)):
            p = bproc.object.create_primitive("PLANE")
            p.set_scale((0.03, 0.03, 0.03))

            if face_no in [0, 1]:
                p.set_rotation_euler([0,np.pi/2,0])
            elif face_no in [2, 3]:
                p.set_rotation_euler([np.pi/2,0,0])

            p.set_location(self.face_centres[face_no])
            points_list.append(p)

        vis_objects = bproc.camera.visible_objects(cam_pose, sqrt_number_of_rays=50)
        objs_vis = vis_objects.intersection(points_list)
        bproc.object.delete_multiple(points_list)
        return len(objs_vis) >= 3

    def generate_environment(self):
        box, _ = self.load_obj("blender_objects/box_tall2.obj", [0, 0, 0], [np.pi / 2, 0, 0], True)
        box.add_uv_mapping("cube", overwrite=True)
        box.clear_materials()

        box.set_scale((0.2, 0.2, 0.2))
        box.set_location([0, 0, 0])
        ceiling, _ = self.load_obj("PLANE", [0, 0, 40], [0, 0, 0])
        ceiling.set_scale((20, 20, 1))
        bproc.lighting.light_surface([ceiling], emission_strength=self.c_params.ambient_light)

        return ceiling, box

    def randomise_env(self, ceiling, box, poi, rand_objs):
        # randomise scene
        bproc.lighting.light_surface([ceiling], emission_strength=np.random.uniform(self.r_params.min_ambient_light,
                                                                                    self.r_params.max_ambient_light))

        self.randomise_material(box)
        light = self.set_spot_light(self.generate_pos_distance_from_point(poi, self.r_params.min_spot_distance,
                                                                          self.r_params.max_spot_distance, 360),
                                    np.random.uniform(self.r_params.min_spot_light, self.r_params.max_spot_light))

        for obj in rand_objs:
            self.randomise_material(obj)
        
        return light

    def rot2eul(self, R):
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        return np.array((alpha, beta, gamma))

    def get_keypoints(self, cam_loc, cam_rot, bb):
        keypoints = np.empty((len(bb), 3))
        for i, point in enumerate(bb):
            distance, _, _ = self.generate_data(end=point, start=cam_loc, start_orient=np.eye(3), end_orient=cam_rot)
            keypoints[i] = distance
        return keypoints

    def add_random_object(self, curr_objs, obj_widths):
        random_objects = ["CUBE", "CYLINDER", "CONE", "SPHERE"]
        obj = bproc.object.create_primitive(np.random.choice(random_objects))

        scale = np.random.random() * self.r_params.obj_size_range + self.r_params.min_obj_size
        random_loc = [0,0,0]

        colliding = True

        while colliding:
            random_loc = bproc.sampler.disk(center=[0,0,scale], radius=self.r_params.distractor_objects_radius)
            colliding=False
            for i in range(len(curr_objs)):
                m_dist = abs(random_loc[0]-curr_objs[i][0]) + abs(random_loc[1]-curr_objs[i][1])
                print("i: ", i)
                print("distance: ", m_dist, obj_widths[i], scale)
                if m_dist < obj_widths[i]*2+scale*2:
                    colliding=True
                    break
            if colliding:
                continue
            break
        obj.set_scale([scale for _ in range(3)] )
        obj.set_location(random_loc)
        obj.set_rotation_euler([0,0,np.random.random()*2*np.pi])
        
        return obj, random_loc, scale

    def setup_scene(self, blue_mat = None, orange_mat = None, green_mat=None):
        # load canonical scene
        cube, poi = self.load_obj("CUBE", [0, 0, 0.1], [0, 0, 0])
        cube.add_uv_mapping("cube", overwrite=True)
        cube.set_scale((0.1, 0.1, 0.1))
        cube.clear_materials()

        self.face_centres = self.get_face_centres(cube)

        print("CUBE BB: ", cube.get_bound_box())

        ceiling, box = self.generate_environment()

        if orange_mat == None:
            orange_mat = bproc.material.create("orange_mat")
            orange_texture = bproc.loader.load_texture(self.orange_mat)
            orange_mat.infuse_texture(orange_texture[0], mode='set', texture_scale=1)
        cube.add_material(orange_mat)

        if blue_mat == None:
            blue_mat = bproc.material.create("blue_mat")
            blue_texture = bproc.loader.load_texture(self.blue_mat)
            blue_mat.infuse_texture(blue_texture[0], mode='set', texture_scale=1)

        if green_mat == None:
            green_mat = bproc.material.create("green_mat")
            green_texture = bproc.loader.load_texture(self.green_mat)
            green_mat.infuse_texture(green_texture[0], mode='set', texture_scale=1)

        return cube, poi, ceiling, box, blue_mat, orange_mat, green_mat

    def render_scene(self, no_images):
        bproc.renderer.enable_depth_output(activate_antialiasing=True)
        bproc.camera.set_intrinsics_from_blender_params(image_height=self.c_params.img_height,
                                                        image_width=self.c_params.img_width)


        cube, poi, ceiling, box, blue_mat, orange_mat, green_mat = self.setup_scene()

        try:
            distances_arr = np.load("%s/pose_info/distances.npy" % self.output_folder)
        except:
            distances_arr = np.empty((5000, 3))
        try:
            rotations_arr = np.load("%s/pose_info/rotations.npy" % self.output_folder)
        except:
            rotations_arr = np.empty((5000, 3, 3))
        try:
            euler_angles_arr = np.load("%s/pose_info/euler_angles.npy" % self.output_folder)
        except:
            euler_angles_arr = np.empty((5000, 3))
        try:
            keypoints_arr = np.load("%s/pose_info/keypoints.npy" % self.output_folder)
        except:
            keypoints_arr = np.empty((5000, 8, 3))


        for i in range(self.start_img, self.start_img+no_images):

            os.environ["BLENDER_PROC_RANDOM_SEED"] = str(i*self.r_params.no_randomisations)
            np.random.seed(i*self.r_params.no_randomisations)

            bproc.utility.reset_keyframes()

            box.clear_materials()
            box.add_material(blue_mat)
            bproc.lighting.light_surface([ceiling], emission_strength=self.c_params.ambient_light)

            rand_objs = []
            obj_locs = [[0,0,0]]
            obj_widths = [0.1]
            for _ in range(self.rand_objs_to_gen):
                rand_obj, obj_loc, obj_width = self.add_random_object(obj_locs, obj_widths)
                rand_obj.clear_materials()
                rand_obj.add_material(green_mat)
                rand_objs.append(rand_obj)
                obj_locs.append(obj_loc)
                obj_widths.append(obj_width)


            cam_pose, visible, cam_loc, cam_rot = self.generate_pos_bb_visible(poi, self.r_params.min_camera_distance,
                                                             self.r_params.max_camera_distance, 90,
                                                             bb=cube.get_bound_box(), obj=cube)

            if not visible:
                i -= 1
                bproc.object.delete_multiple(rand_objs)
                continue

            
            camera_rot = bproc.camera.rotation_from_forward_vec(poi-cam_loc)
            distance, rotation_matrix, euler_angles = self.generate_data(end=poi, start=cam_loc, start_orient=np.eye(3), end_orient=camera_rot)
            keypoints = self.get_keypoints(cam_loc, camera_rot, cube.get_bound_box())

            bproc.camera.add_camera_pose(cam_pose)

            distances_arr[i] = (distance)
            rotations_arr[i] = rotation_matrix
            euler_angles_arr[i] = (euler_angles)
            keypoints_arr[i] = keypoints
            
            # Render the scene
            bproc.renderer.set_max_amount_of_samples(30)
            data = bproc.renderer.render()
            data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

            self.output_file(data, "canonical", i, 0)

            for j in range(self.r_params.no_randomisations):

                bproc.utility.reset_keyframes()

                # randomise environment
                light = self.randomise_env(ceiling, box, poi, rand_objs)

                # Render the randomised scene
                bproc.camera.add_camera_pose(cam_pose)

                data = bproc.renderer.render()
                self.output_file(data, "randomised", i, j)

                light.delete()

            if i % self.save_arrs_interval == 0:
                np.save("%s/pose_info/distances.npy" % self.output_folder, distances_arr)
                np.save("%s/pose_info/rotations.npy" % self.output_folder, rotations_arr)
                np.save("%s/pose_info/euler_angles.npy" % self.output_folder, euler_angles_arr)
                np.save("%s/pose_info/keypoints.npy" % self.output_folder, keypoints_arr)

            bproc.object.delete_multiple(rand_objs)
