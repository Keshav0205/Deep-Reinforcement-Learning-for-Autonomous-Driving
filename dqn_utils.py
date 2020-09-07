import tensorflow as tf
import numpy as np
import carla
import random 
import time 
import queue
import pickle
from PIL import Image
from dqn_functions_extra import map_action, map_from_control, reset_env, process_image, compute_reward, isDone

class Sensors(object):
    '''
    Class for camera and collision sensor and required functions
    '''
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.cam_queue = queue.Queue()
        self.collision_flag = False
        self.lane_crossed = False
        self.lane_crossed_type = ''
        self.camera = self.add_sensor('sensor.camera.rgb')
        self.collision_sensor = self.add_sensor('sensor.other.collision')
        self.lane_sensor = self.add_sensor('sensor.other.lane_invasion')
        self.sensor_list = [self.camera, self.collision_sensor, self.lane_sensor]

        self.camera.listen(lambda image: self.cam_queue.put(image))
        self.collision_sensor.listen(lambda event: self.check_for_collision(event))
        self.lane_sensor.listen(lambda lane_event: self.invasion(lane_event))

    def add_sensor(self, type, tick = '0.0'):
        sensor_bprint = self.world.get_blueprint_library().find(type)

        try:
            sensor_bprint.set_attribute('sensor_tick', tick)
        except:
            pass
        if type == 'sensor.camera.rgb':
            sensor_bprint.set_attribute('image_size_x', '100')
            sensor_bprint.set_attribute('image_size_x', '100')
        
        sensor_transform = carla.Transform(carla.Location(x = 1.5, z = 2.5))
        sensor = self.world.spawn_actor(sensor_bprint, sensor_transform, attach_to = self.vehicle)
        return sensor
    
    def check_for_collision(self, event):
        self.collision_flag = True
    
    def invasion(self, lane_event):
        types = set(x.type for x in lane_event.crossed_lane_markings)
        t = ['%r' % str(x).split([-1]) for x in types]
        self.lane_crossed_type = t[0]
        self.lane_crossed= True
    
    def reset(self):
        self.collision_flag = False
        self.lane_crossed = False
        self.lane_crossed_type = ''
    
    def destroy_sensors(self):
        for sensor in self.sensor_list:
            sensor.destroy()
    

class DQNet():
    '''
    This class represents the model for the Deep Q-learning Network. 
    Input: 84 x 84 RGB images
    Process: 3 Conv nets, 1 Flatten and then divided into 2 streams:-
        1. To calculate the valu function V(s)
        2. To calculate the advantage function A(s,a)
    Finally, an aggregating layer, outputs the Q values for each action, Q(s,a)
    '''
    def __init__(self, state_size, action_space, learning_rate, name):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space) 
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()
        self.learning_rate = learning_rate
        self.name = name
        '''
        the tf.variable_scope will be used to identify which network is currently being used
        the DQN or the target network.
        '''
        with tf.variable_scope(self.name):
            self.inputs = tf.placeholder(tf.float32, [None, *self.state_size], name = 'inputs')
            self.weights = tf.placeholder(tf.float32, [None, 1], name = 'weights')
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name = 'actions')

            # Target network is Reward + gamma * max(Q(s', a'))
            self.target_q = tf.placeholder(tf.float32, [None], name = 'target_metwork')

            '''
            Neural Network definition
            Conv net1: 32 filters, kernel_size = [8,8], strides = [4,], activation = relu, padding = 'valid'
            Conv net1: 64 filters, kernel_size = [4,4], strides = [2,2], activation = relu, padding = 'valid'
            Conv net1: 64 filters, kernel_size = [3,3], strides = [1,1], activation = relu, padding = 'valid'
            Flatten 
            '''

            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs, kernel_size = [8,8],
                strides = [4,4], kernel_initializer = tf.variance_scaling_initializer(),
                padding = 'valid', activation = tf.nn.relu,
                name = 'conv1'
            )

            self.conv2 = tf.layers.conv2d(
                inputs = self.conv1, kernel_size = [4,4],
                strides = [2,2], kernel_initializer = tf.variance_scaling_initializer(),
                padding = 'valid', activation = tf.nn.relu,
                name = 'conv2'
            )

            self.conv3 = tf.layers.conv2d(
                inputs = self.conv2, kernel_size = [3,3],
                strides = [1,1], kernel_initializer = tf.variance_scaling_initializer(),
                padding = 'valid', activation = tf.nn.relu,
                name = 'conv3'
            )

            self.flatten = tf.layers.Flatten(self.conv3)

            self.value_fully = tf.layers.dense(
                inputs = self.flatten, units = 1024, activation = tf.nn.relu,
                kernel_initializer = tf.variance_scaling_initializer(),
                name = 'value_fully'
            )
            self.value = tf.layers.dense(
                inputs = self.value_fully, units = 1, activation = None,
                kernel_initializer = tf.variance_scaling_initializer(),
                name = 'value'
            )

            self.advantage_fully = tf.layers.dense(
                inputs = self.flatten, units = 1024, activation = tf.nn.relu,
                kernel_initializer = tf.variance_scaling_initializer(),
                name = 'advantage_fully'
            )

            self.advantage = tf.layers.dense(
                inputs = self.advantage_fully, units = self.action_size, activation = None,
                kernel_initializer = tf.variance_scaling_initializer(),
                name = 'advantage'
            )

            '''
            Final layer: Q(s,a) = V(s) + A(s,a) - 1/|A| * sum(A(s,a'))
            '''
            self.output = self.value + tf.subtract(self.advantage, 
                                                tf.reduce_mean(self.advantage, axis = 1, keep_dims = True))
            self.output = tf.identity(self.output, name = 'output')
            self.q = tf.reduce_mean(tf.multiply(self.output, self.actions), axis = 1)

            self.abs_errors = tf.abs(self.target_q - self.q)
            self.loss = tf.reduce_mean(self.weights * tf.squared_difference(self.target_q, self.q))
            self.loss_2 = tf.reduce_mean(tf.losses.huber_loss(labels = self.target_q, predictions = self.q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            self.optimizer_2 = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_2)

        def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state):
            exp_tradeoff = np.random.rand()

            explore_prob = explore_stop + (explore_start - explore_start) * np.exp(-decay_rate * decay_step)

            if explore_prob > exp_tradeoff:
                index = random.choice(self.action_size)
                action = self.possible_actions[index]
            
            else:
                Q = sess.run(self.output,feed_dict = {self.inputs: state.reshape((1, *state.shape))})
                index = np.argmax(Q)
                action = self.possible_actions[int(index)]
            
            return index, action, explore_prob


class Sumtree(object):
    '''
    This is a modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.data_pointer = 0
        '''
        number of leaf nodes that contains experiences
        generate the tree with all nodes values =0
        To understand this calculation (2 * capacity - 1) look at the schema above
        Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        Parent nodes = capacity - 1
        Leaf nodes = capacity
        '''
        self.tree = np.zeros(2*self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
    
    def add_data(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[tree_index] = data
        self.update(tree_index, priority)
        if self.data_pointer >= self.capacity:
            ''' Resets and overwrites memory '''
            self.data_pointer = 0
    
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            '''
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
                            0
                           / /
                          1   2
                         / \ / /
                        3  4 5  [6]
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            '''
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self,v):
        '''
        To get the leaf index, priority value fo that leaf and the associated experience
         Tree structure and array storage:
                        Tree index:
                             0         -> storing priority sum
                            / \
                          1     2
                         / \   / \
                        3   4 5   6    -> storing priority for experiences
                        Array type for storing:
                        [0,1,2,3,4,5,6] 
        ''' 
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index -1 
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v-= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]

class Memory(object):
    def __init__(self, capacity, pretrain_length, action_space):
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """

        self.sum_tree = SumTree(capacity)
        self.pretrain_length = pretrain_length
        self.action_space = action_space
        self.action_size = len(action_space)
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()
        # hyperparamters
        self.absolute_error_upper = 1.  # clipped abs error
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001

    def store(self, experience):
        '''
        Stores the experience in the tree with max priority
        During training, the priority is to be adjusted according with the prediction error
        '''
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.sum_tree.add_data(max_priority, experience)
    
    def sample(self, n):
        minibatch = []
        if n > self.sum_tree.capacity:
            print('Sample number more than capacity')
        batch_id = np.empty((n,), dtype=np.int32)
        batch_weights = np.empty((n,1), dtype = np.float32)
        priority_segment = self.sum_tree.total_priority/n

        self.PER_b = np.min([1, self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.sum_tree.tree[-self.sum_tree.capacity:]) / self.sum_tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a, b)
            index, priority, data = self.sum_tree.get_leaf(value)

            sampling_probabilities = priority / self.sum_tree.total_priority
            batch_weights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / self.max_weight
            batch_id = index
            experience = [data]
            minibatch.append(experience)
        
        return batch_id, batch_weights
    
    def batch_update(self, tree_index, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_e)
        for index, p in zip(tree_index, ps):
            self.sum_tree.update(index, p)
    
    def fill_memory(self, map, vehicle, cam_queue, sensors, autopilot = False):
        '''
        This is used to fill up the memory with experiences.
        if autopilot is True, experience is given by it.
        Else, agent takes random actions
        '''
        print('Memory filling initiated')
        reset_env(map, vehicle, sensors)
        if autopilot:
            vehicle.set_autopilot()
        
        for i in range(self.pretrain_length):
            if i%500 == 0:
                print(i, 'Experience Stored')
            state = process_image(cam_queue)
            if autopilot:
                control = vehicle.get_control()
                index = map_from_control(control, self.action_space)
                action = self.possible_actions[index]
            else:
                index = np.random.choice(self.action_size)
                action = self.possible_actions[index]
                control = map_action(index, self.action_space)
                vehicle.apply_controls(control)
            time.sleep(0.25)
            reward = compute_reward(vehicle, sensors)
            done = isDone(reward)
            next_state = process_image(cam_queue)
            experience = state, action, reward, next_state, done
            self.store(experience)
            if done:
                reset_env(map, vehicle, sensors)
            else:
                state = next_state
        print('Memory filling complete: %s experiences stored'%self.pretrain_length)
        vehicle.set_autopilot(enabled = False)
    
    def save_memory(self, filename, object):
        handle = open(filename, 'wb')
        pickle.dump(object, handle)

    def load_memory(self, filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    
