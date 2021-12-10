import matplotlib.pyplot as plt

#env related package
import gym,pickle,time,itertools,argparse
import numpy as np
import pybullet as p
import Embryo

#RL related package
import torch
from Embryo.env.Policy_network import *

#Goal parameters
AI_CELL = 'Cpaaa'
TARGET_CELL = 'ABarpaapp'

#n1: ABarppapp; n2: ABarppppa; n3: ABarppapa, n4: ABarpppap
NEIGHBOR_CANDIDATE_1 = [['ABarppppa', 'ABarppapp'], ['ABarpppap', 'ABarppapp'],
                        ['ABarppapp', 'ABarppapa'], ['ABarppppa', 'ABarpppap']]

#n3: ABarppapa; n4: ABarpppap; n5: ABarppaap; n6: ABarpppaa
NEIGHBOR_CANDIDATE_2 = [['ABarppapa','ABarpppap'],['ABarppapa','ABarppaap'],['ABarppapa','ABarpppaa'],\
                        ['ABarpppap','ABarppaap'], ['ABarpppap','ABarpppaa'], ['ABarppaap','ABarpppaa']]
#n5: ABarppaap; n6: ABarpppaa
NEIGHBOR_CANDIDATE_3 = [['ABarpppaa', 'ABarppaap']]

subgoals = list(itertools.product(NEIGHBOR_CANDIDATE_1, NEIGHBOR_CANDIDATE_2))
# subgoals.append((['ABarppppa', 'ABarppapp', 'ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']))


#Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate, make it larger and try
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 4000

###Pre-define parameters
RENDER_MODE = 'gui'         #render mode: direct or gui


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False 


def demo_run(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--em", type=int, default=0, help="The index of Cpaaa embryo. choose from [0-2]")
    args = parser.parse_args()
    env = gym.make("Embryo-v0", method = RENDER_MODE, embryo_num = args.em)
    dqn = DQN()
    dqn.eval_net.load_state_dict(torch.load('./temp010001000/llmodel_909_succ.pkl', map_location=lambda storage, loc: storage))
    dqn.target_net.load_state_dict(torch.load('./temp010001000/llmodel_909_succ.pkl', map_location=lambda storage, loc: storage))

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

    movement_types = []
    cpaaa_locations = []
    target_locations = []

    print('\nCollecting experience...')

    for i_episode in range(len(subgoals)):
        if i_episode % 150 == 149:
            dqn.e_greedy += 0.05
            if dqn.e_greedy > 0.95:
                dqn.e_greedy = 0.95
        # sg = subgoals[i_episode]
        # env.subgoals[0] = sg[0]
        # env.subgoals[1] = sg[1]
        # env.subgoals[0] = ['ABarppppa', 'ABarppapp']
        # env.subgoals[1] = ['ABarpppap', 'ABarppapa']
        # print('Current Subgoals:')
        # for g in sg:
        #     print(g)
        s = env.reset()
        ep_r = 0
        counter = 0

        while True:
            env.render()
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)             #take action

            # dqn.store_transition(s, a, r, s_)           #store parameters
            counter += 1
            ep_r += r

            if done:
                print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',ep_r)
                cpaaa_locations.append(env.ai_locations)
                target_locations.append(env.target_locations)
                break
            s = s_

        reward_draw += ep_r

        if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY+220:
            reward_list_print.append(reward_draw/10.0)
            episode_list.append(i_episode)


    with open('./cpaaa_locations_predict.pkl', 'wb') as f:
        pickle.dump(cpaaa_locations, f)
    
    with open('./target_locations_predict.pkl', 'wb') as f:
        pickle.dump(target_locations, f)
    
    with open('./episode_reward_list_predict.pkl', 'wb') as f:
        pickle.dump((episode_list,reward_list_print), f)
        

if __name__ == '__main__':
    start = time.time()
    demo_run()
    end = time.time()
    print("\nTraining time : %f " % (end-start))
