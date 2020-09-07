import numpy as np

class h_params(object):
    def __init__(self):
        self.state_size = [84,84,1]
        '''
        Action space = discrete and of the form: (throttle, steer, brake)
        '''
        self.action_space = np.array([
         (0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.25, 0.0),
         (0.5, -0.25, 0.0), (0.5, 0.5, 0.0), (0.5, -0.5, 0.0) 
        ])
        self.learning_rate = 0.005

        self.max_steps = 5000
        self.tota_episodes = 5001
        self.batch_size = 64
        #
        '''
        Fixed Target Q network; tau is the step where the target network is updated.
        '''
        self.max_tau = 5000

        '''
        Parameters for epsilon greedy
        '''
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.00005

        '''
        Q-Learning Hyper params
        pretrain_length: number of experiences to be stored in the memory when initialized
        memory_size: no. of experiences the memory can keep
        load_mem: if True load memory, otherwise fill the memory with new data
        '''
        self.gamma = 0.96
        self.pretrain_length = 100_000
        self.memory_size = 100_000
        self.load_mem = True
        self.memory_load_path = 'reply_memory/memory.pkl'
        self.memory_save_path = 'reply_memory/memory.pkl'

        self.model_save_freq = 10
        self.model_name = 'DQ_Network'
        self.model_path = 'model'
        self.model_save_path = 'model/RL_model.ckpt'
        