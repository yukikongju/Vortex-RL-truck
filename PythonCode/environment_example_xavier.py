#Imports

import sys
vortex_folder = r'C:\CM Labs\Vortex Studio 2020b\bin'
sys.path.append(vortex_folder)

import Vortex as VxSim
import vxatp3 as vxatp
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
import random
import math


#Environment Parameters
WIDTH = 240
HEIGHT = 180
N_OBS = 11

SUB_STEPS = 5

MAX_STEPS = 200


DISTANCE_REWARD = 0.15
SPEED_REWARD = 3
COLLISION_PENALTY = 0.5
CABLE_LENGTH_PENALTY = 1000


class env():

    def __init__(self):

        self.setup_file = '../resources/config/learning_setup.vxc'

        self.scene_file = '../assets/RoughTerrainCrane/Scenes/Learning_Scene/Learning_Scene.vxscene'
        self.load_file = '../assets\Construction\Mechanisms\Objects\Concrete Block/Concrete Block_3500.vxmechanism'

        self.application = vxatp.VxATPConfig.createApplication(self, 'Crane App', self.setup_file)


        #Initialize Action and Observation Spaces for the NN
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1., shape=(N_OBS,), dtype=np.float32)

        self.reward_range = (-2500, 200)

        '''
        # Create a display window
        self.display = VxSim.VxExtensionFactory.create(VxSim.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(VxSim.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.getInput('Viewpoint Name').value = 'Operator Viewpoint'
        self.display.setName('3D Display')
        self.display.getInput(VxSim.DisplayICD.kPlacement).setValue(VxSim.VxVector4(50, 50, 1280, 720))
        self.display.getInput('Physical Position').setValue(VxSim.VxVector3(0.2, 0, 0))
        self.application.add(self.display)
        '''
        #Initialize Scene
        self.initialize_scene()


    def initialize_scene(self):

        # Switch to Editing
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeEditing)


        # load mechanism file
        self.scene = self.application.getSimulationFileManager().loadObject(self.scene_file)
        self.hud_interface = self.scene.findExtensionByName('HUD_VHL')


        # Find Crane and Crane Interface
        self.vxcrane = self.scene.findExtensionByName('RoughTerrainCrane')
        self.crane = VxSim.MechanismInterface(self.vxcrane)
        self.crane_interface = self.crane.findExtensionByName('RL_Interface')
        self.crane_control_interface = self.crane.findExtensionByName('RTC Operator Control Preset')
        self.crane_gameplay_interface = self.crane.findExtensionByName('Gameplay ICD')
        self.crane_display_interface = self.crane.findExtensionByName('Display ICD')
        self.crane_HMI = self.crane.findExtensionByName('Main HMI')
        self.boomtip_texture = self.crane.findExtensionByName('Boomtip Texture')


        # Load load file
        self.vxload = self.scene.findExtensionByName('Concrete Block_3500')
        self.load = VxSim.MechanismInterface(self.vxload)
        self.load_collision_interface = self.load.findExtensionByName('Collision Interface')
        self.load_part = self.load.getAssemblies()[0].getParts()[0]

        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeSimulating)

        # Place load
        load_att_transform = self.crane_interface.getOutputContainer()['Load Attachment Reference'].value
        load_att_translation = VxSim.getTranslation(load_att_transform)

        self.load.inputLocalTransform.value = VxSim.translateTo(load_att_transform,
                                                                VxSim.VxVector3(load_att_translation.x,
                                                                                load_att_translation.y,
                                                                                load_att_translation.z + -1.3))


        # Initialize first key frame
        self.application.update()
        self.keyFrameList = self.application.getContext().getKeyFrameManager().createKeyFrameList("KeyFrameList", False)
        self.application.update()

        self.keyFrameList.saveKeyFrame()
        self.waitForNbKeyFrames(1, self.application, self.keyFrameList)
        self.key_frames_array = self.keyFrameList.getKeyFrames()



    def reset(self):


        self.current_step = 0
        self.reward = 0
        self.max_height = 0

        # Load first key frame
        self.keyFrameList.restore(self.key_frames_array[0])
        self.application.update()

        # Rotate Crane randomly


        # Attach load
        self.load_part.inputControlType.value = VxSim.Part.kControlStatic
        '''
        self.crane_control_interface.getInputContainer()['Hook/Unhook Button'].value = True
        self.application.update()
        self.crane_control_interface.getInputContainer()['Hook/Unhook Button'].value = False
        '''
        while not self.crane_gameplay_interface.getOutputContainer()['Main Riggings Hooked FB'].value:
            self.application.update()

        self.load_part.inputControlType.value = VxSim.Part.kControlDynamic
        for _ in range(60):
            self.application.update()


        # Apply random velocity to load
        random_velocity = VxSim.VxVector3(random.randrange(-30, 30, 1)/10, random.randrange(-30, 30, 1)/10, 0)
        self.load_part.inputLinearVelocity.value = random_velocity


        # Simulate a random number of steps (0-2s)
        for _ in range(random.randrange(0, 120, 1)):
            self.application.update()

        # Remove Swing Brake
        self.crane_HMI.getOutputContainer()['SwingBrake_Switch'].value = False

        # Increase Throttle
        self.crane_HMI.getOutputContainer()['EngineThrottle_Dial'].value = 10

        return self._get_obs()

    def step(self, actions): #takes a numpy array as input
        done = False

        #Apply actions
        i = 0
        for action in actions:
            self.crane_interface.getInputContainer()['input '+str(i)].value = float(action)
            i += 1

        #Step the simulation
        for _ in range(SUB_STEPS):
            self.application.update()

        #Observations
        obs = self._get_obs()

        #rewards
        reward = 0
        boomtip_position = VxSim.getTranslation(self.crane_interface.getOutputContainer()['Boom Tip Reference'].value)
        load_position = VxSim.getTranslation(self.crane_interface.getOutputContainer()['Load Attachment Reference'].value)
        load_speed = self.load_part.outputLinearVelocity.value
        load_collision = self.load_collision_interface.getOutputContainer()['Anything Collision Detected'].value

        distance = math.dist([boomtip_position.x, boomtip_position.y], [load_position.x, load_position.y])
        speed = math.dist([load_speed.x, load_speed.y, load_speed.z], [0, 0, 0])

        #Rewards

        reward += (1-distance) * DISTANCE_REWARD
        reward += (0.6-speed) * SPEED_REWARD
        reward += - COLLISION_PENALTY * load_collision

        self.current_step += 1
        if self.current_step >= MAX_STEPS:
            done = True
        elif self.crane_display_interface.getOutputContainer()['Main Winch Cable 2 POL Length'].value > 105:
            done = True
            reward += - CABLE_LENGTH_PENALTY

        info = {}

        return obs, reward, done, info

    def _get_obs(self):

        #texture = VxSim.Texture.dynamicCast(self.boomtip_texture.getExtension())
        #image = texture.getImage()
        #width = image.getWidth()
        #height = image.getHeight()
        #data = bytes(image.getImageBytes())

        #pilImage = PIL.Image.frombytes("RGB", (width, height), data)
        #pilImage = pilImage.transpose(PIL.Image.FLIP_TOP_BOTTOM)  # Image is flipped vertically.
        #pilImage = pilImage.convert('L')  # Convert to Grayscale
        #pilImage = pilImage.resize((int(width/2), int(height/2)))

        #Crane Positions reduced to {-1, 1}

        obs = []
        obs.append(self.crane_display_interface.getOutputContainer()['Swing Angle'].value/(2*3.14159)) #Crane Angle
        obs.append(self.crane_display_interface.getOutputContainer()['Main Winch Cable 2 POL Length'].value/100.0) #Main Winch length

        turret_position_transformation = self.crane_interface.getOutputContainer()['Turret Reference'].value.inverse_orthogonal()
        boomtip_position = VxSim.getTranslation(turret_position_transformation * self.crane_interface.getOutputContainer()['Boom Tip Reference'].value)
        load_position = VxSim.getTranslation(turret_position_transformation * self.crane_interface.getOutputContainer()['Load Attachment Reference'].value)
        load_speed = self.load_part.outputLinearVelocity.value


        for coordinate in boomtip_position:
            obs.append(coordinate/30.0)

        for coordinate in load_position:
            obs.append(coordinate/30.0)

        for coordinate in load_speed:
            obs.append(coordinate / 5.0)

        return np.array(obs)


    def render(self,  active=True, sync=False):
        #Find current list of displays
        current_displays = self.application.findExtensionsByName('3D Display')

        #If active, add a display and activate Vsync
        if active:
            if len(current_displays) == 0:
                self.application.add(self.display)

            if sync:
                self.application.setSyncMode(VxSim.kSyncSoftwareAndVSync)
            else:
                self.application.setSyncMode(VxSim.kSyncNone)

        #If not, remove the current display and deactivate Vsync
        else:
            if len(current_displays) == 1:
                self.application.remove(current_displays[0])
            self.application.setSyncMode(VxSim.kSyncNone)


    def waitForNbKeyFrames(self,expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter

    def deadband(self,input,deadband):
        if input <= deadband and input >= -deadband:
            return 0
        else:
            return input

