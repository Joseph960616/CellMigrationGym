import matplotlib.pyplot as plt

#env related package
import gym,pickle,time,itertools
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


#n1: ABarppapp; n2: ABarppppa; n3: ABarppapa, n4: ABarpppap
NEIGHBOR_CANDIDATE_1 = [['ABarppppa', 'ABarppapp'], ['ABarpppap', 'ABarppapp'],
                        ['ABarppapp', 'ABarppapa'], ['ABarppppa', 'ABarpppap']]

#n3: ABarppapa; n4: ABarpppap; n5: ABarppaap; n6: ABarpppaa
NEIGHBOR_CANDIDATE_2 = [['ABarppapa','ABarpppap'],['ABarppapa','ABarppaap'],['ABarppapa','ABarpppaa'],\
                        ['ABarpppap','ABarppaap'], ['ABarpppap','ABarpppaa'], ['ABarppaap','ABarpppaa']]
#n5: ABarppaap; n6: ABarpppaa
NEIGHBOR_CANDIDATE_3 = [['ABarpppaa', 'ABarppaap']]
subgoals = list(itertools.product(NEIGHBOR_CANDIDATE_1, NEIGHBOR_CANDIDATE_2))

###Pre-define parameters
RENDER_MODE = 'direct'         #render mode: direct or gui

test_goal = [['ABarppppa', 'ABarppapp'],['ABarpppap', 'ABarppapa'],['ABarpppaa', 'ABarppaap']]
# test_goal = [['ABarpppap', 'ABarppapa'],['ABarpppaa', 'ABarppaap']]
# test_goal = [['ABarppppa', 'ABarppapp', 'ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']]
def demo_run(): 
    env = gym.make("Embryo-v0",method = RENDER_MODE,embryo_num = 0)
    dqn = DQN()

    plt.ion()
    episode_list = []
    reward_list = []
    episode_loss_list = []
    loss = -1
    reward_draw = 0

    action_value_list = []
    episode_action_value_list = []
    episode_subgoal = []
    episode_subgoal_done_step = []
    epi_suc = []
    movement_types = []
    cpaaa_locations = []
    target_locations = []

    print('\nCollecting experience...')

    for i_episode in range(2001):
        action_value_list = []
        loss_list = []
        if i_episode % 150 == 149:
            dqn.e_greedy += 0.05
            if dqn.e_greedy > 0.95:
                dqn.e_greedy = 0.95
        # sg_index = np.random.randint(len(subgoals))
        # s = env.reset(subgoals = subgoals[sg_index])
        s = env.reset(subgoals = test_goal)
        ep_r = 0
        counter = 0

        while True:
            # env.render()
            if i_episode % 1000 == 0 and i_episode != 0:
                name = 'llmodel_' + str(i_episode) + '.pkl'
                torch.save(dqn.eval_net.state_dict(), name)

            a = dqn.choose_action(s)
            action_value_list.append(a)
            s_, r, done, sg_done = env.step(a)             #take action
            t = env.ticks

            dqn.store_transition(s, a, r, s_)           #store parameters
            counter += 1
            ep_r += r
            # print('Step reward: x=', r)
            if dqn.memory_counter > MEMORY_CAPACITY:
                loss = dqn.learn()                      #learn
                if done:
                    # print('Ending reward:', r)
                    print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',ep_r,'\n')
                    if r == 100:
                        epi_suc.append(i_episode)
                        # if i_episode > 900:
                        #     name = 'llmodel_' + str(i_episode) + '_succ.pkl'
                        #     torch.save(dqn.eval_net.state_dict(), name)
            loss_list.append(loss)

            if done:
                cpaaa_locations.append(env.ai_locations)
                target_locations.append(env.target_locations)
                episode_action_value_list.append(action_value_list)
                episode_subgoal_done_step.append(env.subgoal_done_step)
                episode_subgoal.append(env.subgoals)
                episode_loss_list.append(loss_list)
                reward_list.append(ep_r)
                episode_list.append(i_episode)
                break
            s = s_

        # reward_draw += ep_r

        # if i_episode % 10 == 0 and dqn.memory_counter > MEMORY_CAPACITY:
        #     reward_list.append(reward_draw/10.0)
        #     episode_list2.append(i_episode)
        #     reward_draw = 0


    with open('./cpaaa_locations.pkl', 'wb') as f:
        pickle.dump(cpaaa_locations, f)
    
    with open('./target_locations.pkl', 'wb') as f:
        pickle.dump(target_locations, f)
    
    with open('./episode_reward_list.pkl', 'wb') as f:
        pickle.dump((episode_list,reward_list), f)

    with open('./epi_suc.pkl', 'wb') as f:
        pickle.dump(epi_suc, f)

    with open('./episode_action_value_list.pkl', 'wb') as f:
        pickle.dump(episode_action_value_list, f)
    
    with open('./episode_subgoal_done_step.pkl', 'wb') as f:
        pickle.dump(episode_subgoal_done_step, f)

    with open('./episode_subgoal.pkl', 'wb') as f:
        pickle.dump(episode_subgoal, f)
    
    with open('./episode_loss_list.pkl', 'wb') as f:
        pickle.dump(episode_loss_list, f)

if __name__ == '__main__':
    start = time.time()
    demo_run()
    end = time.time()
    print("\nTraining time : %f " % (end-start))
