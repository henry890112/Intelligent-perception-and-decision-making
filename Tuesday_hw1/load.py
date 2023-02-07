import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import open3d
import copy
import os
from math import tan

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "apartment_0/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):  #得到深度圖
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agentf
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE


 ########MY CODE#####
    rgb_rotate_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_rotate_sensor_spec.uuid = "color_rotate_sensor"
    rgb_rotate_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_rotate_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_rotate_sensor_spec.position = [0.0, settings["sensor_height"], -1.5]
    rgb_rotate_sensor_spec.orientation = [
        -np.pi/2,
        0.0,
        0.0,
    ] 
    rgb_rotate_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
###############################

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, rgb_rotate_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
input_floor = input("Input the floor:")   # set the floor

agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, int(input_floor), 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
SCREENSHOT="q"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print(" q for screenshot")
print("#############################")

keystroke = cv2.waitKey(0)


def navigateAndSee(action=""):
    global x, y, z ,rw, rx, ry, rz
   
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)

        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("RGB_rotate", transform_rgb_bgr(observations["color_rotate_sensor"]))
       
        cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        x, y, z ,rw, rx, ry, rz = sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z
    return observations["color_sensor"], observations["color_rotate_sensor"], observations["depth_sensor"], x, y, z

            
action = "move_forward"
navigateAndSee(action)
a = []
photo_of_number = 1

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        a = navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        a = navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        a = navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        fp = open("number"+str(input_floor)+".txt", "w")
        fp.write(str(photo_of_number-1))
        fp.close()
        break

    ###My Code###
    elif keystroke == ord(SCREENSHOT):
        print("action: SCREENSHOT", photo_of_number)

        if not (os.path.exists("./front"+str(input_floor))):
            os.makedirs("front"+str(input_floor))
        if not (os.path.exists("./top"+str(input_floor))):
            os.makedirs("top"+str(input_floor))
        if not (os.path.exists("./depth"+str(input_floor))):
            os.makedirs("depth"+str(input_floor))

        cv2.imwrite("./front"+str(input_floor)+"/front_view_path"+str(photo_of_number)+".png", transform_rgb_bgr(a[0]))
        cv2.imwrite("./top"+str(input_floor)+"/top_view_path"+str(photo_of_number)+".png", transform_rgb_bgr(a[1]))
        cv2.imwrite("./depth"+str(input_floor)+"/depth_path"+str(photo_of_number)+".png", transform_depth(a[2]))
        # Current_pose.append([x, y, z ,rw, rx, ry, rz])
        photo_of_number = photo_of_number + 1
        if int(input_floor) == 0:
            with open('Current_position0.txt', 'a') as fp:           
                data = fp.write(str(x)+" ")
                data = fp.write(str(y)+" ")
                data = fp.write(str(z)+" "+"\n")
        else:
            with open('Current_position1.txt', 'a') as fp:           
                data = fp.write(str(x)+" ")
                data = fp.write(str(y)+" ")
                data = fp.write(str(z)+" "+"\n")
            
    else:
        print("INVALID KEY")
        continue

print("Done for saving images")






        # if (i == 2):
        #     mesh = open3d.geometry.TriangleMesh.create_coordinate_frame()
        #     T = np.eye(4)
        #     T[:3, :3] = mesh.get_rotation_matrix_from_quaternion((Current_pose[1][3]-Current_pose[0][3], 
        #                                                         Current_pose[1][4]-Current_pose[0][4],
        #                                                         Current_pose[1][5]-Current_pose[0][5], 
        #                                                         Current_pose[1][6]-Current_pose[0][6]))
        #     T[0, 3] = Current_pose[1][0]-Current_pose[0][0]
        #     T[1, 3] = Current_pose[1][1]-Current_pose[0][1]
        #     T[2, 3] = Current_pose[1][2]-Current_pose[0][2]

        #     print(T)
        #     mesh_t = copy.deepcopy(mesh).transform(T)
    ###########