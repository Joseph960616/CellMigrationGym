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
    parser.add_argument("--em", type=int, default=0, help="The index of Cpaaa embryo. choose from [0-2]")
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
        import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#env related package
import gym,pickle,time,itertools,argparse
import numpy as np
import pybullet as p
import Embryo

#RL related package
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

subgoals = list(itertools.product(NEIGHBOR_CANDIDATE_1, NEIGHBOR_CANDIDATE_2,NEIGHBOR_CANDIDATE_3))
# subgoals.append((['ABarppppa', 'ABarppapp', 'ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']))


#Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate, make it larger and try
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 8000

LL_MODEL_PATH = currentdir + '/trained_models/hdrl_llmodel.pkl'

###Pre-define parameters
RENDER_MODE = 'gui'         #render mode: direct or gui

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False 

# test_goal = [['ABarppppa', 'ABarppapp'],['ABarpppap', 'ABarppapa'],['ABarpppaa', 'ABarppaap']]
def demo_run(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--em", type=int, default=0, help="The index of Cpaaa embryo. choose from [0-2]")
    args = parser.parse_args()
    env = gym.make("Embryo-v0", method = RENDER_MODE, embryo_num = args.em)
    dqn = DQN()
    # dqn.eval_net.load_state_dict(torch.load(LL_MODEL_PATH, map_location=lambda storage, loc: storage))
    # dqn.target_net.load_state_dict(torch.load(LL_MODEL_PATH, map_location=lambda storage, loc: storage))

    episode_list = []
    reward_list_print = []
    reward_draw = 0
    episode_subgoal_done_step = []

    cpaaa_locations = []
    target_locations = []

    for i_episode in range(len(subgoals)):
        print("Episode:",i_episode)
        s = env.reset(subgoals = subgoals[i_episode])
        while True:
            env.render()
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)             #take action
        
            if done:
                print('Episode {} done in {} steps\n'.format(i_episode,env.ticks))
                cpaaa_locations.append(env.ai_locations)
                target_locations.append(env.target_locations)
                episode_subgoal_done_step.append(env.subgoal_done_step)
                break
            s = s_

    # with open('./cpaaa_locations_predict.pkl', 'wb') as f:
    #     pickle.dump(cpaaa_locations, f)
    
    # with open('./target_locations_predict.pkl', 'wb') as f:
    #     pickle.dump(target_locations, f)

if __name__ == '__main__':
    start = time.time()
    demo_run()
    end = time.time()
    print("\nTotal simulation time : %f " % (end-start))

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

            if done:
                print('Episode:{} done in {} steps\n'.format(i_episode,env.ticks))
                cpaaa_locations.append(env.ai_locations)
                target_locations.append(env.target_locations)
                break
            s = s_


    # with open('./cpaaa_locations_predict.pkl', 'wb') as f:
    #     pickle.dump(cpaaa_locations, f)
    
    # with open('./target_locations_predict.pkl', 'wb') as f:
    #     pickle.dump(target_locations, f)

if __name__ == '__main__':
    start = time.time()
    demo_run()
    end = time.time()
    print("\nTotal simulation time : %f " % (end-start))
