import matplotlib.pyplot as plt

#env related package
import gym,pickle,time
import numpy as np
import pybullet as p
import Embryo

#RL related package
import torch
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN
from Embryo.env.Policy_network import *

#Goal parameters
AI_CELL = 'Cpaaa'
TARGET_CELL = 'ABarpaapp'

NEIGHBOR_CANDIDATE_1 = [['ABarppppa', 'ABarppapp'], ['ABarpppap', 'ABarppapp'],
                        ['ABarppppa', 'ABarppapp', 'ABarpppap'], ['ABarppapp', 'ABarppapa', 'ABarppppa']]
NEIGHBOR_CANDIDATE_2 = [['ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppaap'], ['ABarpppaa', 'ABarppapa'], ['ABarpppaa', 'ABarppaap'], 
                        ['ABarpppap', 'ABarppapa', 'ABarpppaa'], ['ABarpppap', 'ABarppapa', 'ABarppaap'], 
                        ['ABarpppaa', 'ABarppaap', 'ABarpppap'], ['ABarpppaa', 'ABarppaap', 'ABarppapa'],
                        ['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']]
NEIGHBOR_CANDIDATE_3 = [['ABarpppaa', 'ABarppaap']]

#Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate, make it larger and try
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 8000

###Pre-define parameters
RENDER_MODE = 'direct'         #render mode: direct or gui


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False 


def demo_run(): 
    env = gym.make("Embryo-v0",method = RENDER_MODE)
    dqn = DQN()
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

    episode_action_value_list = []
    action_value_total = 0

    state = []
    episode_state = []

    epi_suc = []
    movement_types = []
    cpaaa_locations = []
    target_locations = []

    print('\nCollecting experience...')

    for i_episode in range(1001):
        action_value_list = []
        if i_episode % 150 == 149:
            dqn.e_greedy += 0.05
            if dqn.e_greedy > 0.95:
                dqn.e_greedy = 0.95
        s = env.reset()
        ep_r = 0
        counter = 0

        while True:
            # env.render()
            if i_episode % 1000 == 0:
                name = 'llmodel_' + str(i_episode) + '.pkl'
                torch.save(dqn.eval_net.state_dict(), name)

            # state.append(s)
            a = dqn.choose_action(s)
            action_value_list.append(a)
            s_, r, done, info = env.step(a)             #take action

            dqn.store_transition(s, a, r, s_)           #store parameters
            counter += 1
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                loss = dqn.learn()                      #learn
                if done:
                    print('Single episode reward: %f' % r)
                    print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',ep_r,'\n')
                    if r == 1000:
                        epi_suc.append(i_episode)
                        if i_episode > 900:
                            name = 'llmodel_' + str(i_episode) + '_succ.pkl'
                            torch.save(dqn.eval_net.state_dict(), name)

            if done:
                cpaaa_locations.append(env.ai_locations)
                target_locations.append(env.target_locations)
                episode_action_value_list.append(action_value_list)
                # episode_state.append(state)
                break
            s = s_

        reward_draw += ep_r

        if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY+220:
            reward_list_print.append(reward_draw/10.0)
            episode_list.append(i_episode)


    with open('./cpaaa_locations.pkl', 'wb') as f:
        pickle.dump(cpaaa_locations, f)
    
    with open('./target_locations.pkl', 'wb') as f:
        pickle.dump(target_locations, f)
    
    with open('./episode_reward_list.pkl', 'wb') as f:
        pickle.dump((episode_list,reward_list_print), f)
    
    with open('./epi_suc.pkl', 'wb') as f:
        pickle.dump(epi_suc, f)

    with open('./episode_action_value_list.pkl', 'wb') as f:
        pickle.dump(episode_action_value_list, f)

    # with open('./episode_state.pkl', 'wb') as f:
    #     pickle.dump(episode_state, f)
        

if __name__ == '__main__':
    start = time.time()
    demo_run()
    end = time.time()
    print("\nTotal training time : %f " % (end-start))
