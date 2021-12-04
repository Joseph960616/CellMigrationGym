#add parent dir to find package
import os, inspect, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import matplotlib.pyplot as plt

#env related package
import gym
import numpy as np
import pybullet as p

#RL related package
import torch
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN
from Embroy.env.Policy_network import DQN, Net
import Embroy

#Goal parameters
AI_CELL = 'Cpaaa'
TARGET_CELL = 'ABarpaapp'

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 8000

###Pre-define parameters
RUN_LEARNING = True
START_POINT = 168
END_POINT = 190
TICK_RESOLUTION = 10
TIME_STEP = 60/TICK_RESOLUTION
episode = 40
#render mode: direct or gui or TCP(not working yet)
RENDER_MODE = 'gui'

######ignore boundary model, pass first

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False 


def demo_run(): 
    env = gym.make("Embroy-v0")
    dqn = DQN()
    # model = DQN()
    # model = DQN(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=11000)
    # model.save("deepq_embryo")
    # print('\nModel saved to local')
    # del model # remove to demonstrate saving and loading
    # print('\nLoading model from local...')
    # model = DQN.load("deepq_embryo")

    fig1 = plt.figure(1)
    plt.ion()
    episode_list = []
    reward_list = []
    reward_list_print = []
    loss_list = []
    episode_loss_list = []
    reward_draw = 0
    loss = -1
    loss_total = 0

    action_value_list = []
    episode_action_value_list = []
    action_value_total = 0
    multiple = True

    print('\nCollecting experience...')

    for i_episode in range(4000):
        if i_episode % 150 == 149:
            dqn.e_greedy += 0.05
            if dqn.e_greedy > 0.95:
                dqn.e_greedy = 0.95
        #Choosing different embryo for multiple embryo model training
        if multiple == True:
            embryo = np.random.randint(5)
            s = env.reset(embryo)
        else:
            s = env.reset()
        ep_r = 0
        counter = 0

        while True:
            if i_episode % 1000 == 0:
                name = 'dqn_eval_net_' + str(i_episode) + '.pkl'
                torch.save(dqn.eval_net.state_dict(), name)
            # print("\n\n state:")
            # print(s)
            a = dqn.choose_action(s)

            # take action
            s_, r, done = env.step(a)

            dqn.store_transition(s, a, r, s_)
            counter += 1
            ep_r += r

            if dqn.memory_counter > MEMORY_CAPACITY:
                loss = dqn.learn()
                if done:
                    env.render()
                    print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',ep_r)

            if done:
                break
            s = s_

        reward_draw += ep_r

        if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY+220:
            reward_list_print.append(reward_draw/10.0)

        if i_episode % 10 == 0:
            episode_list.append(i_episode)
            reward_list.append(reward_draw/10.0)
            action_value_list.append(action_value_total/10.0)
            plt.figure(1)
            plt.cla()
            plt.plot(episode_list, reward_list, label='Reward')
            plt.xlabel('Training Epochs')
            plt.ylabel('Reward')
            plt.draw()
            plt.pause(0.1)

            reward_draw = 0
            action_value_total = 0

            if i_episode % 1000 == 0 and i_episode > 0:
                fig1.savefig('fig_reward_'+str(i_episode)+'.eps', format='eps', dpi=fig1.dpi)


# def demo_run_stable(): 
#     reward_list = []
#     distances_to_target = []
#     action_value_list = []
#     episode_action_value_list = []
#     action_value_total = 0

#     env = gym.make("Embroy-v0")
#     print('\nCollecting experience...')
#     ###still use Zi's policy
#     # model = DQN()
#     model = DQN(MlpPolicy, env, verbose=1)
#     model.learn(total_timesteps=11000)
#     model.save("deepq_embryo")
#     print('\nModel saved to local')
#     del model # remove to demonstrate saving and loading
#     print('\nLoading model from local...')
#     model = DQN.load("deepq_embryo")

#     obs = env.reset()

#     if RENDER_MODE == 'gui':
#         gui = True
#     else:
#         gui = False

#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info, distance_to_target = env.step(action)
#         distances_to_target.append(distance_to_target)
#         reward_list.append(rewards)
#         print('Rewards:',rewards)
#         if dones:
#             env.render(gui, start = env.start_point,stop = env.end_point)
#             break
#         env.close()
    
#     ###Draw rewards figure
#     fig2 = plt.figure()
#     plt.xlabel("Training Epochs")
#     plt.ylabel("Reward")
#     plt.plot([ k for k in range(len(distances_to_target))],distances_to_target)
#     plt.savefig('rewards.eps', bbox_inches='tight')

    # ###Draw distance figure
    # fig = plt.figure(1)
    # plt.title("Distance to target cell over time")
    # plt.xlabel("Time(s)")
    # plt.ylabel("Distance to target cell(um)")
    # plt.plot([ k for k in range(len(distances_to_target))],distances_to_target)
    # plt.savefig('Embryo_distance.png', bbox_inches='tight')

# env = gym.make("Embroy-v0")
# from random import random,choice
# reward = 0
# for i in range(100):
#     num = 100 * choice((-1,1)) *random()
#     reward += num
#     time.sleep(1)
#     env.render(stage_num = env.end_point - env.start_point)
#     print("total reward:" + str(reward))
#     p.resetSimulation()
        

if __name__ == '__main__':
    demo_run()
