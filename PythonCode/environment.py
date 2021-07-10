
# Imports
import sys
import csv
import time
import math
import random
import Vortex
import vxatp3
import numpy as np

# Add Vortex to the system path to be able to use Vortex and vxatp3
vortex_folder = r'C:\CM Labs\Vortex Studio 2020b\bin'
sys.path.append(vortex_folder)

class env():
    def __init__(self):
        self.empty_spot_var = None

        """File Locations"""
        # Setup File
        self.setup_file = "../VortexFiles/SetupFile/EmptySceneSetup.vxc"
        # self.actual_truck_file = "C:/Users/rl-truck-summer-2021/Desktop/Truck_V3/mandate-main/Assets/Mechanisms/Semi Truck - Trailer/Semi Truck - Single.vxmechanism"
        # Scene Files
        self.scene_file = "../VortexFiles/EmptyScene/NoTruckScene.vxscene"
        # Truck Files
        self.dummy_truck_file = "../VortexFiles/DummyTruck/Mechanism/DummyTruck.vxmechanism"
        self.actual_truck_file = "C:/Users/rl-truck-summer-2021/Desktop/Truck_V3/mandate-main/Assets/Mechanisms/Semi Truck - Trailer/Semi Truck - Single.vxmechanism"

        """Setup The Simulation and Load Related Files"""
        # Create the app
        self.application = vxatp3.VxATPConfig.createApplication(self, "Truck Simulation", self.setup_file)
        # self.vx_actual_truck = self.application.getSimulationFileManager().loadObject(self.actual_truck_file)
        # self.actual_truck = Vortex.MechanismInterface(self.vx_actual_truck)
        # Load the scene
        self.vx_scene = self.application.getSimulationFileManager().loadObject(self.scene_file)
        self.scene = Vortex.SceneInterface(self.vx_scene)
        # Load actual Truck
        self.vx_actual_truck = self.application.getSimulationFileManager().loadObject(self.actual_truck_file)
        self.actual_truck = Vortex.MechanismInterface(self.vx_actual_truck)
        # Load 19 dummy trucks
        self.dummy_trucks = []
        for _ in range(19):
            current_truck = self.application.getSimulationFileManager().loadObject(self.dummy_truck_file)
            self.dummy_trucks.append(Vortex.MechanismInterface(current_truck))

        """Load VHL Files"""
        # VHL to control truck movement
        self.truck_interface = self.actual_truck.findExtensionByName("Vehicle Interface")
        # VHL to get truck observations
        self.observation_interface = self.actual_truck.findExtensionByName("Output Observations")

        self.initialization()


    def get_parking_pose(self):
        # Import csv with spot poses [x, y, z, degree_rx, degree_ry, degree_rz]
        csv_output = np.genfromtxt('example.csv', delimiter=' ')

        all_poses = []
        for line in csv_output:
            x, y, z, drx, dry, drz = line
            rx = np.deg2rad(drx)
            ry = np.deg2rad(dry)
            rz = np.deg2rad(drz)
            # Construct transformation matrix [Rotation Matrix, translation vector; zero vector, 1]
            current_pose_m = Vortex.rotateTo(Vortex.createTranslation(x, y, z), Vortex.VxVector3(rx, ry, rz))
            all_poses.append(current_pose_m)
        return all_poses

    def initialization(self):
        # Go into editing mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

        # Find all parking poses - list of transformation matrices
        all_parking_poses = self.get_parking_pose()

        """Randomly choose empty parking spot"""
        # Choose empty spot location
        empty_spot = random.randint(0, len(all_parking_poses)-1)
        empty_spot_location = all_parking_poses[empty_spot]
        self.empty_spot_var = empty_spot_location
        del all_parking_poses[empty_spot]

        # Put dummy trucks in all 19 locations that are not the empty spot
        for index, truck in enumerate(self.dummy_trucks):
            position = all_parking_poses[index]
            truck.inputLocalTransform.value = position

        """Randomly choose Truck Starting Position"""
        # Define area where truck can start in [top, bottom, left, right]
        start_bounds = [70, 2, -25, 0]
        # Define truck dimensions from truck origin [front, left, back, right]
        truck_dimensions = [4.75, 1.5, -14.5, -1.5]

        # Find random (x, y, theta) such that truck starts within start area
        inside = False
        while not inside:
            candidate_x = random.uniform(-24, -2)
            candidate_y = random.uniform(4, 68)
            candidate_angle = random.uniform(0, 2 * np.pi)
            candidate_truck_position = [candidate_x, candidate_y, candidate_angle]
            # Check pose is within start area
            inside = self.check_within_bounds(candidate_truck_position, start_bounds, truck_dimensions)

        # Put actual truck in random starting position
        new_position = Vortex.createTranslation(candidate_x, candidate_y, 0.3)
        current_pose_matrix = Vortex.rotateTo(new_position, Vortex.VxVector3(0, 0, candidate_angle))
        self.actual_truck.inputLocalTransform.value = current_pose_matrix

        """Start simulation"""
        # Switch to simulation
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

        """Save this initial keyframe"""
        self.application.update()
        self.keyFrameList = self.application.getContext().getKeyFrameManager().createKeyFrameList("KeyFrameList", False)
        self.application.update()
        self.keyFrameList.saveKeyFrame()
        self.waitForNbKeyFrames(1, self.application, self.keyFrameList)
        self.key_frames_array = self.keyFrameList.getKeyFrames()

    def reset(self):
        # Load first key frame
        self.keyFrameList.restore(self.key_frames_array[0])
        self.application.update()

        # Go into editing mode
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

        # Find all parking poses - list of transformation matrices
        all_parking_poses = self.get_parking_pose()

        """Randomly choose empty parking spot"""
        # Choose empty spot location
        empty_spot = random.randint(0, len(all_parking_poses) - 1)
        empty_spot_location = all_parking_poses[empty_spot]
        self.empty_spot_var = empty_spot_location
        del all_parking_poses[empty_spot]

        # Put dummy trucks in all 19 locations that are not the empty spot
        for index, truck in enumerate(self.dummy_trucks):
            position = all_parking_poses[index]
            truck.inputLocalTransform.value = position

        """Randomly choose Truck Starting Position"""
        # Define area where truck can start in [top, bottom, left, right]
        start_bounds = [70, 2, -25, 0]
        # Define truck dimensions from truck origin [front, left, back, right]
        truck_dimensions = [4.75, 1.5, -14.5, -1.5]

        # Find random (x, y, theta) such that truck starts within start area
        inside = False
        while not inside:
            candidate_x = random.uniform(-24, -2)
            candidate_y = random.uniform(4, 68)
            candidate_angle = random.uniform(0, 2 * np.pi)
            candidate_truck_position = [candidate_x, candidate_y, candidate_angle]
            # Check pose is within start area
            inside = self.check_within_bounds(candidate_truck_position, start_bounds, truck_dimensions)

        # Put actual truck in random starting position
        new_position = Vortex.createTranslation(candidate_x, candidate_y, 0.3)
        current_pose_matrix = Vortex.rotateTo(new_position, Vortex.VxVector3(0, 0, candidate_angle))
        self.actual_truck.inputLocalTransform.value = current_pose_matrix

        """Start simulation"""
        # Switch to simulation
        vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)


    def waitForNbKeyFrames(self,expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter

    def check_within_bounds(self, truck_initial_position, start_bounds, truck_dim):
        """
        Function to check if truck is within start area
        :param truck_initial_position: list of [x, y, theta] representing truck pose
        :param start_bounds: list of [top, bottom, left, right] representing start area [max_y, min_y, min_x, max_x]
        :param truck_dim: list of [front, left, back, right] representing truck dim from truck origin
        :return: True or False on whether truck is inside start area
        """
        # Decompose truck position and start area
        x, y, angle = truck_initial_position
        top, bottom, left, right = start_bounds

        # Find truck corners
        truck_corners = self.find_truck_extremities(x, y, angle, truck_dim)

        # If all four corners are inside start area, then truck is inside start area
        corners_within = 0
        for corner in truck_corners:
            x_within = self.check_middle(corner[0], left, right)
            y_within = self.check_middle(corner[1], bottom, top)
            if x_within and y_within:
                corners_within += 1
        if corners_within == 4:
            return True
        else:
            return False

    def find_truck_extremities(self, x, y, angle, truck_dim):
        """
        Finds the corners of the truck given location of truck
        :param x: Truck's x location
        :param y: Truck's y location
        :param angle: Truck's angle from x axis
        :param truck_dim: [front, left, back, right]
        :return: Corners [bottom left, bottom right, top right, top left]
        """
        # Find corner angles and lengths of truck
        angles = self.angles_from_truck_dim(truck_dim)
        lengths = self.corner_distance_from_truck_dim(truck_dim)

        # Find new corners based on translation and rotation applied
        all_new_points = []
        for i in range(4):
            new_x = self.find_new_x(x, lengths[i], angle, angles[i])
            new_y = self.find_new_y(y, lengths[i], angle, angles[i])
            new_point = (new_x, new_y)
            all_new_points.append(new_point)
        all_new_points = sorted(all_new_points, key=lambda tup: (tup[1], tup[0]))
        return all_new_points

    def angles_from_truck_dim(self, truck_dim):
        [pos_x, pos_y, neg_x, neg_y] = truck_dim
        theta = math.atan2(pos_y, pos_x)
        alpha = math.atan2(pos_y, neg_x)
        beta = math.atan2(neg_y, neg_x)
        gamma = math.atan2(neg_y, pos_x)
        angles = [theta, alpha, beta, gamma]
        return angles

    def corner_distance_from_truck_dim(self, truck_dim):
        [pos_x, pos_y, neg_x, neg_y] = truck_dim
        one = math.sqrt(pos_x ** 2 + pos_y ** 2)
        two = math.sqrt(neg_x ** 2 + pos_y ** 2)
        three = math.sqrt(neg_x ** 2 + neg_y ** 2)
        four = math.sqrt(pos_x ** 2 + neg_y ** 2)
        lengths = [one, two, three, four]
        return lengths

    def find_new_x(self, old_x, length, global_angle, local_angle):
        return old_x + length * math.cos(global_angle + local_angle)

    def find_new_y(self, old_y, length, global_angle, local_angle):
        return old_y + length * math.sin(global_angle + local_angle)

    def check_middle(self, value, small, big):
        if small < value < big:
            return True
        else:
            return False

    def move(self, action):
        """
        Move the actual truck based on action given
        :param action: Array = [Engine Running, Throttle, Brake, Steering, Gear]
        :return: Doesn't Return anything
        """
        # Apply actions
        self.truck_interface.getInputContainer()["Engine Running"].value = action[0]
        self.truck_interface.getInputContainer()["Throttle"].value = action[1]
        self.truck_interface.getInputContainer()["Brake"].value = action[2]
        self.truck_interface.getInputContainer()["Steering"].value = action[3]
        self.truck_interface.getInputContainer()["Gear"].value = action[4]

    def run_for_n_steps(self, n_steps):
        """
        Engine - True or False
        Throttle - between 0 and 1
        Brake - between 0 and 1
        Steering - between -1 and 1
        Gear - pos (forward) or neg (backwards) or 0 (neutral)
        """
        for i in range(n_steps):
            self.get_observations()
            self.move([True, 1, 0, 1, 1])
            self.application.update()

    def ray_cast_distance(self, ray_cast_name):
        raycast_object = self.observation_interface.getOutputContainer()[ray_cast_name].value
        intersection_point = raycast_object.getOutput('Intersection Point').value
        ray_origin = raycast_object.getOutput('Ray')['Origin'].value
        p1 = np.array([intersection_point.x, intersection_point.y, intersection_point.z])
        p2 = np.array([ray_origin.x, ray_origin.y, ray_origin.z])
        dist = np.sqrt(np.sum((p1 - p2)**2, axis=0))
        return dist

    def find_distance_to_spot(self, curr_loc, spot_loc):
        # Find distance
        curr_trans = Vortex.getTranslation(curr_loc)
        spot_trans = Vortex.getTranslation(spot_loc)
        p1 = np.array([curr_trans.x, curr_trans.y, curr_trans.z])
        p2 = np.array([spot_trans.x, spot_trans.y, spot_trans.z])
        distance = np.sqrt(np.sum((p1 - p2)**2, axis=0))

        # Find angle
        curr_angle_rz = Vortex.getRotation(curr_loc)[2]
        spot_angle_rz = Vortex.getRotation(spot_loc)[2]
        angle = curr_angle_rz - spot_angle_rz

        return distance, angle

    def get_observations(self):
        # Pose Information
        truck_world_pose = self.observation_interface.getOutputContainer()["Truck Pose"].value
        hinge_angle = self.observation_interface.getOutputContainer()["Hinge Angle"].value
        trailer_world_pose = self.observation_interface.getOutputContainer()["Trailer Pose"].value  # May not need

        # Distance to Spot output = [distance, angle]
        distance_to_spot, angle_to_spot = self.find_distance_to_spot(truck_world_pose, self.empty_spot_var)

        # Velocity Information
        truck_linear_vel = self.observation_interface.getOutputContainer()["Truck Linear Velocity"].value
        truck_angular_vel = self.observation_interface.getOutputContainer()["Truck Angular Velocity"].value
        # print(truck_linear_vel, truck_angular_vel)
        hinge_velocity = self.observation_interface.getOutputContainer()["Hinge Velocity"].value
        trailer_linear_vel = self.observation_interface.getOutputContainer()["Trailer Linear Velocity"].value
        trailer_angular_vel = self.observation_interface.getOutputContainer()["Trailer Angular Velocity"].value

        # RayCast Information
        truck_front_rc = self.ray_cast_distance("Raycast Truck Front")
        truck_front_right_rc = self.ray_cast_distance("Raycast Truck Front Right")
        truck_front_left_rc = self.ray_cast_distance("Raycast Truck Front Left")
        truck_right_rc = self.ray_cast_distance("Raycast Truck Right")
        truck_left_rc = self.ray_cast_distance("Raycast Truck Left")
        trailer_left_rc = self.ray_cast_distance("Raycast Trailer Left")
        trailer_right_rc = self.ray_cast_distance("Raycast Trailer Right")
        trailer_back_rc = self.ray_cast_distance("Raycast Trailer Back")
        trailer_back_left_rc = self.ray_cast_distance("Raycast Trailer Back Left")
        trailer_back_right_rc = self.ray_cast_distance("Raycast Trailer Back Right")

        # Truck Collision
        intersection_sensor = self.observation_interface.getOutputContainer()['Truck Collided'].value
        if intersection_sensor:
            print("Collision Detected")
            self.reset()
            print("Done Re-initializing")



env = env()
env.run_for_n_steps(800)