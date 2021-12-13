import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
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

#Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate, make it larger and try
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 4000

###Pre-define parameters
RENDER_MODE = 'gui'         #render mode: direct or gui
DRL_MODEL_PATH = currentdir + '/trained_models/drl_model.pkl'


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False 


def demo_run(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--em", type=int, default=1, help="The index of Cpaaa embryo. choose from [0-2]")
    args = parser.parse_args()
    env = gym.make("Embryo-v0", method = RENDER_MODE, embryo_num = args.em)
    dqn = DQN()
    # dqn.eval_net.load_state_dict(torch.load(DRL_MODEL_PATH, map_location=lambda storage, loc: storage))

    episode_list = []
    reward_list_print = []
    reward_draw = 0
    cpaaa_locations = []
    target_locations = []

    print('\nCollecting experience...')

    for i_episode in range(10):
        if i_episode % 150 == 149:
            dqn.e_greedy += 0.05
            if dqn.e_greedy > 0.95:
                dqn.e_greedy = 0.95
        s = env.reset()
        ep_r = 0

        while True:
            env.render()
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)             #take action
            # dqn.store_transition(s, a, r, s_)           #store parameters
            ep_r += r

            if done:
                print('Episode:', i_episode, 'Done in', env.ticks, 'steps. Reward:',ep_r)
                cpaaa_locations.append(env.ai_locations)
                target_locations.append(env.target_locations)
                break
            s = s_

        reward_draw += ep_r

        if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY+220:
            reward_list_print.append(reward_draw/10.0)
            episode_list.append(i_episode)


    # with open('./cpaaa_locations_predict.pkl', 'wb') as f:
    #     pickle.dump(cpaaa_locations, f)
    
    # with open('./target_locations_predict.pkl', 'wb') as f:
    #     pickle.dump(target_locations, f)

if __name__ == '__main__':
    start = time.time()
    demo_run()
    end = time.time()
    print("\nTotal simulation time : %f " % (end-start))
