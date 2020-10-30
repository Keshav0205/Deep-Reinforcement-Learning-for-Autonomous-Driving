import sys
sys.path.append('..')
import manual_control
import carla
import time 
import random
import logging
import pygame
from multiprocessing import Process
import numpy as np
import tensorflow as tf
import queue
import dqn_utils
from dqn_utils import DQNet, Sumtree, Memory
from dqn_functions_extra import map_action, isDone, reset_env, process_image, compute_reward, get_split_batch
from dqn_params import h_params

def render(clock, world, display):
    clock.tick_busy_loop(30) #max client fps
    world.tick(clock)
    world.render(display)
    pygame.display.flip()

def update_target_graph():
    '''
    To copy the parameters of DQN to target network
    Credits: Arthur Juliani https://github.com/awjuliani
    '''
    from_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DQNetwork')
    to_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TargetNetwork')
    op_holder = []

    for from_, to_ in zip(from_, to_):
        holder.append(to_.assign(from_))
    
    return holder
def init_tf():
    '''
    Tensorflow initializer
    '''
    cp = tf.ConfigProto()
    tf.reset_default_graph()
    return cp

def train_loop(hyper_params, vehicle, map, sensors):

    cp = init_tf()
    dqn = DQNet(hyper_params.state_size,
    hyper_params.action_space, hyper_params.learning_rate,
    name = hyper_params.model_name)
    t_net = DQNet(hyper_params.state_size,
    hyper_params.action_space, hyper_params.learning_rate,
    name = 'TargetNetwork')
    writer = tf.summary.FileWriter('Summary')
    tf.summary.scalar('Loss', DQNet.loss)
    tf.summary.scalar('Hubor_Loss', DQNet.loss_2)
    tf.histogram('Weights', DQNet.weights)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    memory = Memory(hyper_params.memory_size, hyper_params.pretrain_length, 
    hyper_params.action_space)
    if hyper_params.load_mem:
        memory = memory.load_memory(hyper_params.memory_load_path)
        print('Memory Loaded')
    else:
        memory.fill_memory(map, vehicle, sensors.cam_queue, sensors, autopilot=True)
        memory.save_memory(hyper_params.memory_save_path, memory)

    with tf.session(config = cp) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        m = 0
        decay_step = 0
        tau = 0

        update_t = update_target_graph()
        sess.run(update_t)
        for episode in range(1,hyper_params.total_episodes):
            reset_env(map, vehicle, sensors)
            state = process_image(sensors.cam_queue)
            done = False
            start = time.time()
            episode_reward = 0

            if hyper_params.model_save_freq:
                if episode % hyper_params.model_save_freq == 0:
                    save_path = saver.save(sess, hyper_params.model_save_path)
            
            for step in range(hyper_params.max_steps):
                tau += 1
                decay_step += 1
                index, action, explore_prob = DQNet.predict_action(sess, hyper_params.explore_start,
                 hyper_params.explore_stop, hyper_params.decay_rate, decay_step, state)

                car_controls = map_action(index, hyper_params.action_space)
                vehicle.apply_control(car_controls)
                time.sleep(0.25)
                next_state = process_image(sensors.cam_queue)
                reward = compute_reward(vehicle, sensors)
                episode_reward += reward 
                done = isDone(reward)

                experience = state, action, reward, next_state, done
                memory.store(experience)

                tree_index, batch, weights = memory.sample(hyper_params.batch_size)
                states, actions, rewards, next_states, dones = get_split_batch(batch)

                q_next = sess.run(DQNet.output, feed_dict = {DQNet.inputs : next_states})
                q_target_next = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs: next_states})
                
                q_target_batch = []
                for i in range(0, len(dones)):
                    terminal = dones[i]
                    action = np.argmax(q_next[i])
                    if terminal:
                        q_target_batch.append(rewards[i])
                    else:
                        t = rewards[i] + ( hyper_params.gamma * q_target_next[i][action] )
                         q_target_batch.append(t)
                
                targets = np.array([each for each in q_target_batch])

                _,_, loss, loss_2, abs_erros = sess.run([DQNet.optimizer, 
                DQNet.optimizer_2, DQNet.loss, DQNet.loss_2],
                feed_dict = {
                    DQNet.inputs : states,
                    DQNet.target_q : targets,
                    DQNet.actions: actions,
                    DQNet.weights: weights})
                
                memory.batch_update(tree_index, abs_errors)

                if tau > hyper_params.max_tau:
                    update_t = update_target_graph()
                    sess.run(update_target)
                    m+=1
                    tau=0
                    print('Model Updated!')
                state = next_state
                if done:
                    print('{} episode finished. Total reward: {}'.format(episode, episode_reward))
                    break

def test_loop(hyper_params, vehicle, map, sensors):
    cp = init_tf()
    with tf.session(config = cp) as sess:

        saver = tf.train.import_meta_graph(hyper_params.model_save_path + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(hyper_params.model_path))

        if saver is None:
            print("Didn't load")
        
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name(hyper_params.model_name + '/inputs:0')
        output = graph.get_tensor_by_name(hyper_params.model_name + '/output:0')

        episode_reward = 0

        while True:
            state = process_image(sensors.cam_queue)
            Qs = sess.run(output, feed_dict = {inputs: state.reshape((1, *state.shape))})
            index = np.argmax(Qs)
            
            car_controls = map_action(index, hyper_params.action_space)
            vehicle.apply_control(car_controls)
            reward = compute_reward(vehicle, sensors)

            episode_reward += reward

            done = isDone(reward)

            if done:
                print("Episode end, reward: {}".format(episode_reward))
                reset_env(map, vehicle, sensors)
                episode_reward = 0
            
            else:
                time.sleep(0.25)

def control_loop(vehicle_id, host, port, test_flag):
    actor_list = []
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        Map = world.get_map()
        vehicle = next((x for x in world.get_actors() if x.id == vehicle_id), None)
        sensors = dqn_utils.Sensors(world, vehicle)
        hyper_params = h_params()

        if test_flag:
            test_loop(hyper_params, vehicle, Map, sensors)
        else:
            train_loop(hyper_params, vehicle, map, sensors)
        
    finally:
        sensors.destroy_sensors()
    
def render_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF) 
        hud = manual_control.HUD(args.width, args.height)
        world = manual_control.World(client.get_world(), hud, args.filter, args.rolename)
        p = Process(target=control_loop, args = (world.player.id, args.host, args.port, args.test, )) 
        p.start()

        clock = pygame.time.Clock()

        while True:
            render(clock, world, display)

    finally:
        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
        pygame.quit()
    

def main():
    argparser = argparse.ArgumentParser(
        description='DEEP_RL_CARLA')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.audi.tt")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
    