# Deep-Reinforcement-Learning-for-Autonomous-Driving
This repository contains my implementation of Deep Reinforcement Learning using Deep Neural Networks for autonomous vehicle's control and behaviorial navigation tested using the CARLA Simulator. </br>
1. **dqn_main_agent.py**: This Python file is the central one which calls all the functions required for he CARLA simulator to run and the training and testing of the agent.  
2. **dqn_utils.py**: This Python file contains the classes - Sensors, DQNetwork, Memory and Sumtree. These are required for creating, modifying and adding/deletig the memory instances along with the Deep Neural Network.
3. **dqn_functions_extra.py**: This Python file contains some auxilliary functions for the training and other operations. For the sake of easy access and reduced complexity and length of the code, I have collected all the functions here.
4. **dqn_params.py**: This Python file contains the hyperparameters related to the memory buffer, neural network and the reinforcement learning agent.
5. __State-space__ : The state is simply the 84 x 84 RGB image captured by the on-board vehicle camera which is processed by the neural network.
6. __Action-space__ : It is an array of tuples of the form (throttle, steering, brake). The output of the neural network is mapped to one of the tuples in this array using the minimum of the Euclidean norm calculated with respect to the obained value from the neural network.
7. __Reward__ : The reward is computed in real-time using the wheel odometry, collision, lane-invasion sensor values and is discounted over an episode with a preset discounted factor.
8. __Neural Network Architecture__: 84 x 84 RGB image -> Convolution_1-> Convolution_2 -> Convolution_3 -> Flatten -> Fully Connected.
9. The output of the __Flatten layer__ is divided into the next __Fully Connected Layer__ and an Advantage output.
10. __Final Layer__: Q(s,a) = V(s) + A(s,a) - 1/|A| * sum(A(s,a'))
11. [Link to project description](https://drive.google.com/file/d/1eVmQC4MBqDEFZle5uc2Wb4PD-Jtt8TC_/view?usp=sharing)
</br>
Note: Simulation results to be added soon.
</br>

