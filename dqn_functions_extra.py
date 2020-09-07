import numpy as np
import carla
import time
from PIL import Image

def get_split_batch(batch):
    states = np.array([each[0][0] for each in batch], ndmin=3)
    actions = np.array([each[0][1] for each in batch])
    rewards = np.array([each[0][2] for each in batch])
    next_states = np.array([each[0][3] for each in batch], ndmin=3)
    dones = np.array([each[0][4] for each in batch])

    return states, actions, rewards, next_states, dones

def map_action(action, action_space):
    '''
    for mapping discrete action to actual car's control
    '''
    control = carla.VehicleControl()
    control_seq = action_space[action]
    control.throttle = control_seq[0]
    control.steer = control_seq[1]
    control.brake = control_seq[2]
    return control

def map_from_control(control, action_space):
    '''
    for mapping continuous carla controls to discrete action values
    '''
    control_vec = np.array([control.throttle, control.steer, control.brake])
    dist = []
    for c in action_space:
        dist.append(np.linalg.norm(c - control_vec))
    ret = np.argmin(dist)
    return ret

def reset_env(map, vehicle, sensors):
    vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 1.0))
    time.sleep(1)
    spawn_points = map.get_spawn_points()
    spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
    vehicle.set_transform(spawn_point)
    time.sleep(1.5)
    sensors.reset_sensors()

def process_image(queue):
    image = queue.get()
    arr = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    arr = np.reshape(arr, [image.height, image.width])
    arr = arr[:,:,:,3]
    arr = arr[:,:,:,-1]
    image = Image.fromarray(arr).convert('L')
    image = np.array(image.resize((84,84)))
    image = np.reshape(image, (84,84,1))
    return image

def compute_reward(vehicle, sensors):
    max_speed = 14
    min_speed = 2
    velocity = vehicle.get_velocity
    vehicle_speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

    speed_reward = (abs(vehicle_speed -min_speed)) / (max_speed - min_speed)

    lane_reward = 0
    if (vehicle_speed > max_speed) or (vehicle_speed< min_speed):
        speed_reward = -0.04
    if sensors.lane_crossed:
        if sensors.lane_crossed_type == 'Broken' or sensors.lane_crossed_type == 'None': 
            lane_reward -= 0.5
            sensors.lane_crossed = False

    if sensors.collison_flag:
        return -1
    else:
        speed_reward + lane_reward

def isDone(reward):
    if reward <=-1:
        return True
    else:
        return False
