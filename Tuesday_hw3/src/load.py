import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math
from scipy.io import loadmat

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "./apartment_0/apartment_0/habitat/mesh_semantic.ply"
path = "./apartment_0/apartment_0/habitat/info_semantic.json"

# load colors to make the color correct
colors = loadmat('color101.mat')['colors']
colors = np.insert(colors, 0, values=np.array([[0,0,0]]), axis=0) #to make the color be correct

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

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

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(colors.flatten())
    semantic_img.putdata(semantic_obs.flatten().astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
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

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
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

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.01) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])


RRT_position = np.load('./RRT_position.npy')
print(RRT_position[0][1])
print(RRT_position[0][0])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([RRT_position[0][1], 0.0, RRT_position[0][0]+0.01])  # agent in world space
agent.set_state(agent_state)


# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


target_color = {'refrigerator':(255, 0, 0), 'rack':(0, 255, 133), 'cushion':(255, 9, 92), 'lamp':(160, 150, 20), 'cooktop':(7, 255, 224)} #圓括號是tuple不可更改 方括號是list做更改
Input_target = input("Target name: ")
numer_of_step = 1
def navigateAndSee(action=""):
    global numer_of_step, x, y, z  # 為何這裡要global還沒搞清楚

    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)
        rgb_img = transform_rgb_bgr(observations["color_sensor"])
        semantic_img = transform_semantic(id_to_label[observations["semantic_sensor"]])

        cv2.imshow("RGB", rgb_img)
        #cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        cv2.imshow("semantic", semantic_img)
        chosen_color = np.where((semantic_img[:,:,2] == target_color[Input_target][0])* (semantic_img[:,:,1] == target_color[Input_target][1])* (semantic_img[:,:,0] == target_color[Input_target][2]))
        if len(chosen_color[0])==0:
            cv2.imwrite("./my_data/image"+str(numer_of_step)+".png", rgb_img)
        
        # opencv為bgr但上面給定為rgb
        chosen_color = np.where((semantic_img[:,:,2] == target_color[Input_target][0])* (semantic_img[:,:,1] == target_color[Input_target][1])* (semantic_img[:,:,0] == target_color[Input_target][2]))
        # 將指定部份半透明化且當list中有數值才會進入imshow堧如果妹有此判斷式再沒有照到指定物體時會報錯
        if len(chosen_color[0])!=0:
            rgb_img[chosen_color] = cv2.addWeighted(rgb_img[chosen_color], 0.6, semantic_img[chosen_color], 0.4, 50)
            cv2.imshow("semantic2", rgb_img)
        cv2.imwrite("./my_data/image"+str(numer_of_step)+".png", rgb_img)

        numer_of_step+=1
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        x, y, z  = sensor_state.position[0],sensor_state.position[1],sensor_state.position[2]
    return x, y, z

def dis_and_angel(first, second, third):
    current_vector = second - first
    next_vector = third - second
    print(current_vector)
    print(current_vector.shape)

    current_norm = np.linalg.norm(current_vector)
    next_norm = np.linalg.norm(next_vector)
    a_dot_b = current_vector.dot(next_vector)
    cos_theta=np.arccos(a_dot_b/(current_norm* next_norm))
    angle = np.rad2deg(cos_theta)
    # use cross to check turn left or turn right
    cross_value = current_vector[0]* next_vector[1] - current_vector[1]* next_vector[0]
    if cross_value > 0:
        rotate_action = "turn_left"
    else:
        rotate_action = "turn_right"
    return current_norm, angle, rotate_action


action = "move_forward"
navigateAndSee(action)
# our RRT position is (z, x) so if I want to got the (0, 0, -1)direction by this must follow my set
current_norm, angle, rotate_action = dis_and_angel(np.array([RRT_position[0][0] + 1, RRT_position[0][1]]), RRT_position[0], RRT_position[1])
angle_step = int(angle)
for i in range(angle_step):
    cv2.waitKey(0)
    navigateAndSee(rotate_action)  

# append RRT's shape because of the end point to use the dis_and_angel must have 3 point
# can use to find the target roientation by define the last point
print(RRT_position)
RRT_position = np.append(RRT_position, values = [RRT_position[-1]], axis = 0)
print(RRT_position) 

x, y, z = navigateAndSee()
for step in range(RRT_position.shape[0]-1):
 
    current_norm, angle, rotate_action = dis_and_angel(RRT_position[step], RRT_position[step+1], RRT_position[step+2])
    forward_step = int(current_norm // 0.01)  # 取整數可以知道要走幾步
    for i in range(forward_step):
        cv2.waitKey(0)
        navigateAndSee("move_forward")
    # print(RRT_position.shape[0]-1)
    # print(step)
    if ((RRT_position.shape[0]-3)==step):
        break
    for j in range(int(angle)):
        cv2.waitKey(0)
        navigateAndSee(rotate_action)


path = "./video/" + Input_target + ".mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(path, fourcc, 100, (512, 512))
print(numer_of_step)
for i in range(1, numer_of_step):
    img = cv2.imread("./my_data/image"+str(i)+".png")
    videowriter.write(img)
videowriter.release()
