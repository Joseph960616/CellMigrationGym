import os,time,inspect, datetime
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
projectdir = os.path.dirname(parentdir)

os.sys.path.insert(0, parentdir)

import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pybullet as p
import gym
from Embryo.env.Embryo_agent import EmbryoAgent
from Embryo.utils.utils import rgb2gray, gray2twobit, depth_conversion
from Embryo.utils.embryo import Embryo

NEIGHBOR_MODEL_PATH = parentdir + '/trained_models/neighbor_model.pkl'
DRL_MODEL_PATH = parentdir + '/trained_models/drl_model.pkl'

#Goal parameters
AI_CELL = 'Cpaaa'
TARGET_CELL = 'ABarpaapp'

STATE_CELL_LIST = ['ABarppaap', 'ABarppapa', 'ABarppapp', 'ABarpppaa', 'ABarpppap', 'ABarppppa',\
                    'ABarppppp', 'ABprapaaa', 'ABprapaap', 'ABprapapa', 'ABprapapp', 'Caaaa','Caaap',\
                    'Cpaap', 'Epla', 'Eplp', 'Epra', 'Eprp']

#Pre-define/calculated parameters
PLANE_RESOLUTION = 0.254
RADIUS_SCALE_FACTOR = 1.0
DISTANCE_TRANSPARENT = 40
np.random.seed(0)
# EMBRYO_VOLUME_LIST = [2500578.08321, 2818131.28451, 2334542.95732, 3890650.18134,\
#                         2762373.40873, 2477879.24970, 2653539.39170, 6728081.76875,\
#                         2327138.18813, 2220084.34469]
EMBRYO_VOLUME_LIST = [2653539.39170,2327138.18813,2220084.34469]


#RL related parameters
SUB_GOAL_TOLERANCE = 5              #need to be tune for different embryo
AI_SPEED_PER_MIN = 60                #need to be tune for different embryo
AI_BEGIN_REWARD = 70
TICK_RESOLUTION = 10

#Render related parameters
TIME_STEP = 60/TICK_RESOLUTION

class EmbryoBulletEnv(gym.Env):
    metadata = {'render.modes': ['human']} 
    def __init__(self, method='direct', embryo_num = 0):
        #Initialization
        print('\nInitializing environment...')
        self.data_path = projectdir + '/data/cpaaa_%d/nuclei/t%03d-nuclei'
        print("self.data_path ",self.data_path % (0,0))
        self.start_point = 0
        self.end_point = 0
        self.ticks = 0
        self.stage = 0
        self.embryo_num = embryo_num
        self.set_seed = self.seed(0)
        self.radius_scale_factor = RADIUS_SCALE_FACTOR
        self.state_cell_list = STATE_CELL_LIST
        self.embryo_volume_list = EMBRYO_VOLUME_LIST
        self.plane_resolution = PLANE_RESOLUTION
        self.tick_resolution = TICK_RESOLUTION
        self.state_value_dict = {}
        self.neighbor_model = pickle.load(open(NEIGHBOR_MODEL_PATH, 'rb'))
        self.drl_model = pickle.load(open(DRL_MODEL_PATH, 'rb'))
        self.create(method)
        #Load the world
        self.cell = p.loadMJCF(os.path.join(parentdir,'Embryo/utils/cell_neighbour.xml'))
        self.agent = p.loadMJCF(os.path.join(parentdir,'Embryo/utils/agent.xml'))

        ########## Reinforcement Learning Agent ##############
        self.ai = EmbryoAgent(self.client,self.agent[0],AI_CELL)
        self.target_cell_name = TARGET_CELL
        self.ai_locations = []
        self.target_locations = []
        ########################################################

        #gym space
        self.n_observations = (len(self.state_cell_list) + 1) * 3
        self.actions = self.ai.action_list
        self.n_actions = len(self.actions)

        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.box.Box(
                                low = np.array([0,0,0]*(len(self.state_cell_list) + 1), dtype=np.float32),
                                high = np.array([300,300,100]*(len(self.state_cell_list) + 1), dtype=np.float32))

        #Calculating Embryo volume
        # em = Embryo(projectdir+'/data/cpaaa_%d/nuclei/')
        # em.read_data()
        # em.get_volume()
        # self.embryo_volume = em.volume
        ##pre-calculated embryo volume based on above procedure (saving time)
        self.embryo_volume = self.embryo_volume_list[self.embryo_num]
        #Load nuclei data from 1-200
        self.data_dicts = self.load_data()
        self.end_tick = (self.end_point - self.start_point) * self.tick_resolution - 11
        print("start point: %d\nend point: %d\n" %(self.start_point,self.end_point ))
    

        #Data interpolation
        self.pos_interpolations_a,self.cell_name_interpolations_a = \
            self.interpolation(self.cell_name_a, self.pos_a,self.tick_resolution)
        self.pos_interpolations_neighbour_a,self.cell_name_interpolations_neighbour_a = \
            self.interpolation(self.cell_name_neighbour_a, self.pos_neighbour_a,self.tick_resolution)
        self.pos_interpolations_target_a,self.cell_name_interpolations_target_a = \
            self.interpolation(self.cell_name_target_a, self.pos_target_a,self.tick_resolution)

        self.state_cell_list = self.cell_name_interpolations_neighbour_a[0][:,0]

        #Reward related parameters
        self.neighbor_goal_counter = 0
        self.neighbor_goal_achieved_num = 40
        ai2traget_dist_begin = np.linalg.norm(self.pos_interpolations_target_a[0][0,0] - \
                                                    self.pos_interpolations_target_a[0][1,0])
        ai2traget_dist_end = np.linalg.norm(self.pos_interpolations_target_a[-1][0,0] - \
                                                    self.pos_interpolations_target_a[-1][1,0])
        print("AI to target begining distance: %f\nAI reached target distance: %f" %\
        (ai2traget_dist_begin,ai2traget_dist_end))
        
        self.ai_begin_reward_dist = ai2traget_dist_begin * 0.9
        self.ai_target_tolerance = ai2traget_dist_end * 1.1
        self.ai.speed_base = (ai2traget_dist_begin - ai2traget_dist_end) / (self.end_tick * 2)
        # print(self.ai_begin_reward_dist,self.ai_target_tolerance)
        print("AI base speed: %f" % self.ai.speed_base)


        p.resetDebugVisualizerCamera(cameraDistance=80, \
                                cameraYaw=0, \
                                cameraPitch=-90.1, \
                                cameraTargetPosition=self.pos_interpolations_target_a[0][0,0])

        print('\nInitialization finished')
    
    def create(self, method):
        """
        establish connection between pybullet and client
        input:
            method(str): gui/direct
        """
        print('\nConnecting to pybullet client...')
        if method == "gui":
            self.client = p.connect(p.SHARED_MEMORY)
            if (self.client < 0):
                self.client = p.connect(p.GUI)
        elif method == "direct":
            self.client = p.connect(p.DIRECT)
        p.setTimeStep(TIME_STEP)
        p.setRealTimeSimulation(0)
        print('\nConnected to pybullet client with %s render mode'% method)

    def volume_ratio(self, cn):
        """
        Get cell's volume ratio
        """
        if cn[0:2]=="AB":
            v=0.55*(0.5**(len(cn)-2))
        elif cn=="P1":
            v=0.45
        elif cn=="EMS":
            v=0.45*0.54
        elif cn=="P2":
            v=0.45*0.46
        elif cn[0:2]=="MS":
            v=0.45*0.54*0.5*(0.5**(len(cn)-2))
        elif cn=="E":
            v=0.45*0.54*0.5
        elif cn[0]=="E" and len(cn)>=2 and cn[1] != "M":
            v=0.45*0.54*0.5*(0.5**(len(cn)-1))
        elif cn[0]=="C":
            v=0.45*0.46*0.53*(0.5**(len(cn)-1))
        elif cn=="P3":
            v=0.45*0.46*0.47
        elif cn[0]=="D":
            v=0.45*0.46*0.47*0.52*(0.5**(len(cn)-1))
        elif cn=="P4":
            v=0.45*0.46*0.47*0.48
        elif cn in ['Z2', 'Z3']:
            v=0.45*0.46*0.47*0.48*0.5
        else:
            # print('ERROR!!!!! CELL NOT FOUND!!!!', cn)
            v=0.00000001

        return v

    def radius_ratio(self, cn):
        """
        Get cell's radius ratio
        """
        v = self.volume_ratio(cn)
        return v**(1.0/3)

    def get_radius(self, cn):
        """
        Get cell's actual volume
        """
        v = self.volume_ratio(cn)
        radius = pow(self.embryo_volume * v / (4 / 3.0 * np.pi), 1/3.0)
        radius = radius * self.radius_scale_factor

        return radius


    def load_data(self, start = 1, end = 200):
        """
        Load simluation data
        input:
            start(int):start stage of nuclei data
            end(int): end stage of nuclei data
        Return:
            data_dicts(list): each element represent a data dictionary with 
                            cell's name as key, tuple (position, nuclei_radius) as value
                            Eg. stage 170 with cell name 'Cpaaa':
                                data_dicts[170]['Cpaaa'] -->([240.0, 230.0, 70.0], 15.0)
        """
        data_path = self.data_path
        state_cell_list = self.state_cell_list
        target_cell_name = self.target_cell_name
        ai_cell_name = self.ai.name
        stage_num = end - start + 1
        ai_list = []
        target_list = []
        data_dicts = []
        pos_a = []
        radius_a = []
        cell_name_a = []

        pos_neighbour_a = []
        radius_neighbour_a = []
        cell_name_neighbour_a = []

        pos_target_a = []
        radius_target_a = []
        cell_name_target_a = []
        for i in range (stage_num):
            path = data_path % (self.embryo_num,start+i) 
            pos = []
            radius = []
            cell_name = []

            pos_neighbour = []
            radius_neighbour = []
            cell_name_neighbour = []

            pos_target = []
            radius_target = []
            cell_name_target = []
            with open(path) as file:
                for line in file:
                    line = line[:len(line)-1]
                    vec = line.split(', ')
                    if vec[9] == '':
                        continue
                    else:
                        id = int(vec[0])
                        #pixels as unit of each location
                        location = np.array([(float(vec[5])), \
                                            (float(vec[6])), \
                                            (float(vec[7]) / self.plane_resolution)])
                        # #um as unit of each location
                        # location = np.array([(float(vec[5]) * self.plane_resolution), \
                        #                     (float(vec[6]) * self.plane_resolution), \
                        #                     (float(vec[7]))])
                        pos.append(location)
                        #radius needs to multiple resolution when converting to um unit
                        # radius.append(float(vec[8]) / 2)
                        radius.append(self.radius_ratio(vec[9]))
                        cell_name.append(vec[9])

                        if vec[9] in state_cell_list:
                            pos_neighbour.append(location)
                            # radius_neighbour.append(float(vec[8]) / 2)
                            radius_neighbour.append(self.radius_ratio(vec[9]))
                            cell_name_neighbour.append(vec[9])
                        if vec[9] in [ai_cell_name,target_cell_name]:
                            pos_target.append(location)
                            # radius_target.append(float(vec[8]) / 2)
                            radius_target.append(self.radius_ratio(vec[9]))
                            cell_name_target.append(vec[9])
            ###ALL data
            pos_a.append(np.array(pos))
            radius_a.append(np.array(radius))
            cell_name_a.append(np.array(cell_name))            

            ###NEIGHBOUR data
            #sorting according to cell name
            neighbour_index = np.argsort(cell_name_neighbour)
            cell_name_neighbour = np.array(cell_name_neighbour)[neighbour_index]
            pos_neighbour = np.array(pos_neighbour)[neighbour_index]
            radius_neighbour = np.array(radius_neighbour)[neighbour_index]

            pos_neighbour_a.append(pos_neighbour)
            radius_neighbour_a.append(radius_neighbour)
            cell_name_neighbour_a.append(cell_name_neighbour)
            
            ###AGENT and TARGET data
            target_index = np.argsort(cell_name_target)[::-1]
            cell_name_target = np.array(cell_name_target)[target_index]
            pos_target = np.array(pos_target)[target_index]
            radius_target = np.array(radius_target)[target_index]

            pos_target_a.append(np.array(pos_target))
            radius_target_a.append(np.array(radius_target))
            cell_name_target_a.append(np.array(cell_name_target))

            if self.ai.name in cell_name:
                ai_list.append(i+1)
            if TARGET_CELL in cell_name:
                target_list.append(i+1)

            pos_radius = list(zip(pos, radius))
            data_dict = dict(zip(cell_name, pos_radius))
            data_dicts.append(data_dict)
        

        if ai_list != []:
            self.ai_first_appear = ai_list[0]
            self.target_first_appear = target_list[0]
            self.ai_last_appear = ai_list[-1]
            self.target_last_appear = target_list[-1]
        else:
            self.ai_first_appear = 0
            self.ai_first_appear_index = 0
            self.ai_last_appear = 0
            self.target_first_appear = 0
            self.target_last_appear = 0
        
        #find the start/end point of the embryo (starts 15 mins after AI cell born)
        # self.start_point = max(self.ai_first_appear,self.target_first_appear) + 15
        self.start_point = self.ai_first_appear + 15
        self.end_point = min(self.ai_last_appear,self.target_last_appear)
        if self.end_point - self.start_point > 23:
            self.end_point = self.start_point + 23

        #observation data for all cell
        self.pos_a = np.array(pos_a[self.start_point:self.end_point],dtype=object)
        self.radius_a = np.array(radius_a[self.start_point:self.end_point],dtype=object)
        self.cell_name_a = np.array(cell_name_a[self.start_point:self.end_point],dtype=object)
        self.ai_first_appear_index = np.where(self.cell_name_a[0]==self.ai.name)[0][0]

        #observation data for neighbour cell data(sorted according to cell name)
        self.pos_neighbour_a = np.array(pos_neighbour_a[self.start_point:self.end_point],dtype=object)
        self.radius_neighbour_a = np.array(radius_neighbour_a[self.start_point:self.end_point],dtype=object)
        self.cell_name_neighbour_a = np.array(cell_name_neighbour_a[self.start_point:self.end_point],dtype=object)

        #observation data for target cell
        self.pos_target_a = np.array(pos_target_a[self.start_point:self.end_point],dtype=object)
        self.radius_target_a = np.array(radius_target_a[self.start_point:self.end_point],dtype=object)
        self.cell_name_target_a = np.array(cell_name_target_a[self.start_point:self.end_point],dtype=object)

        return data_dicts[self.start_point:self.end_point]

    def interpolation_two_point(self, data_past, data_current,interpolatioin_number):
        """
        Generate interpolation point between two point
        input:
            data_past(np.array): location of one point
            data_current(np.array): location of another point
            interpolatioin_number(int): number of intermediate point to generate

        Return:
            pos(list): list of points inbetween two point
        """
        increment = (data_current- data_past) / interpolatioin_number
        pos = [0] * interpolatioin_number
        pos[0] = data_past
        for i in range(1,10):
            ########### add noise to interpolation location##################
            pos[i] = pos[i-1] + increment + np.random.normal(0, 0.1, 3)
        return pos
    
    def interpolation(self, cell_name_a, pos_a, interpolation_number):
        """
        Generate interpolation point between each stage
        input:
            cell_name_a(list): cell name data set of all stage
            pos_a(list): position data of cell of all stage
            start(int): start stage of the interpolation
            end(int): end stage of the interpolation
            interpolation_number(int): number of point to generate
                                        (including first original point)
        
        Return:
            pos_interpolations_a(np.array) -> eg.pos_interpolations_a[10][:,5] ->\ 
                                            positional data all cell at stage 10, intermediate 5
            cell_name_interpolations_a(np.array)
        """
        pos_interpolations_a = []
        cell_name_interpolations_a = []
        radius_interpolation_a = []
        
        for i in range(1, len(pos_a)):      #one stage
            pos_interpolations = []
            cell_name_interpolations = []
            # radius_interpolations = []
            for j in range(len(cell_name_a[i])):        #one cell in every stage
                if cell_name_a[i][j] in cell_name_a[i - 1]:     #when cell still exist in new stage
                    index_past = np.where(cell_name_a[i - 1]==cell_name_a[i][j])[0][0]  #find where it is for the past cell
                    pos_interpolation = self.interpolation_two_point(pos_a[i - 1][index_past],pos_a[i][j],interpolation_number)
                    cell_name_interpolation = [cell_name_a[i-1][index_past]] * interpolation_number
                    # radius_interpolation = [radius_a[i-1][index_past]] * interpolation_number

                    pos_interpolations.append(pos_interpolation)
                    cell_name_interpolations.append(cell_name_interpolation)
                    # radius_interpolations.append(radius_interpolation)
                else:                                           #when cell divide into two daughter cell
                    pos_interpolation = [pos_a[i][j]] * interpolation_number
                    cell_name_interpolation = [cell_name_a[i][j]] * interpolation_number
                    # radius_interpolation = [radius_a[i][j]] * interpolation_number
                    pos_interpolations.append(pos_interpolation)
                    cell_name_interpolations.append(cell_name_interpolation)
                    # radius_interpolations.append(radius_interpolation)
            
            cell_name_interpolations_a.append(np.array(cell_name_interpolations))
            pos_interpolations_a.append(np.array(pos_interpolations))
            # radius_interpolation_a.append(radius_interpolations)

        return np.array(pos_interpolations_a,dtype=object),\
                np.array(cell_name_interpolations_a,dtype=object)
    
    def get_reward(self):
        """
        Get reward of the movement decision
        """
        r = 0
        done = False
        stage = self.ticks // 10
        timestep = self.ticks % 10
        ai_location = self.ai.get_observation()
        ai_radius = self.get_radius(self.ai.name)
        target_location = self.pos_interpolations_target_a\
                            [stage][1,timestep]
        # print("\nai: {}\ttarget: {}".format(ai_location,target_location))
        #target reach model
        dist2target = np.linalg.norm(target_location - ai_location)
        # print("\ndist2target: {}".format(dist2target))
        if dist2target < self.ai_begin_reward_dist:
            r += (self.ai_begin_reward_dist -  dist2target)
            if dist2target < self.ai_target_tolerance:
                self.neighbor_goal_counter += 1
                if self.neighbor_goal_counter == self.neighbor_goal_achieved_num:
                    done = True
                    if self.ticks < self.end_tick * 0.9:                
                        r = 0
                        print('Target reached too fast!')
                    else:                                               
                        r = 1000
                        print("Target successfully reached!")
                    neighbor_goal_counter = 0
                    return r, done
        # print("target model reward: %f" % r)

        #Pressure model
        for i in range(len(self.cell_name_interpolations_neighbour_a[stage][:,timestep])):
            cell_name = self.cell_name_interpolations_neighbour_a[stage][i,timestep]
            if cell_name != self.ai.name:
                cell_location = self.pos_interpolations_neighbour_a[stage][i,timestep]
                dist = np.linalg.norm(cell_location - ai_location)
                # print("\naidist2cell: {}".format(dist))
                cell_radius = self.get_radius(cell_name)
                sum_radius = cell_radius + ai_radius
                # print("sum_radius: {}".format(sum_radius))
                dead_factor = 0.4
                ok_factor = 0.7
                if dist > ok_factor * sum_radius:
                    r += 0
                elif dist > dead_factor * sum_radius and dist <= ok_factor * sum_radius:
                    r += (ok_factor - float(dist) / sum_radius) / (dead_factor - ok_factor)
                elif dist < dead_factor * sum_radius:
                    print('hit other cell:', cell_name)
                    r = -1000
                    done = True
                    return r, done
        # print("pressure model reward: %f" % r)
        return r, done

    def get_state(self):
        """
        Get state of the obseration cell(positional data)
        """
        stage = self.ticks // 10
        timestep = self.ticks % 10
        s = self.ai.get_observation()
        self.ai_locations.append(s[:3])
        self.target_locations.append(self.pos_interpolations_target_a[stage][1,timestep].tolist())
        for pos in self.pos_interpolations_neighbour_a[stage][:,timestep]:
            s += [round(num,2) for num in pos]
        return s

    def step(self,action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done).
        Args:
            action(int): an action provided by the agent
        Returns:
            s (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            (Not used)info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        sg_done = False
        info = None

        self.ai.apply_action(action)
        # p.stepSimulation()
        s_ = self.get_state()
        r, done = self.get_reward()
        self.ticks += 1
        if self.ticks == self.end_tick:
            print('Time is up! Goal NOT achieved!')
            r = 0
            done = True

        return s_, r, done, info

    def reset(self,embryo_num = 0):
        """
        Returns:
            observation (object): the initial observation.
        """
        #reload the world
        p.resetSimulation(self.client)
        self.cell = p.loadMJCF(os.path.join(parentdir,'Embryo/utils/cell_neighbour.xml'))
        self.agent = p.loadMJCF(os.path.join(parentdir,'Embryo/utils/agent.xml'))
        p.changeVisualShape(self.agent[0],-1,rgbaColor=[1, 0, 0, 1])     #AI cell -> Red
        p.changeVisualShape(self.agent[1],-1,rgbaColor=[0, 1, 0, 1])     #Target cell -> Green
        for ce in self.cell:
            p.changeVisualShape(ce,-1,rgbaColor=[0, 0, 1, 1])       #State cell -> Blue
        p.resetBasePositionAndOrientation(self.agent[0],self.pos_interpolations_target_a[0][0][0],[0,0,0,1])
        #reset pre-define value
        self.ticks = 0
        self.neighbor_goal_counter = 0
        #reset the observation
        self.ai_locations = []
        self.target_locations = []
        s = self.get_state()
        #choose embryo to train for multiple embryo case
        self.embryo_num = embryo_num

        return s

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', img_resolution = 128):
        """
        visualize one timestep of the migrating process using 3D OPENGL rendering when called
            parameter: 
            mode: direct/human
            RED -> AI cell, GREEN -> Target cell, BLUE -> Neighbour cell
        """
        # print(self.cou,self.rand1[self.cou], self.rand2[self.cou])
        agent = self.agent
        cell = self.cell
        stage = self.ticks // 10
        timestep = self.ticks % 10
        orientation = [0,0,0,1]
        img_start = img_resolution // 4 + 1             # cutting between [33,97)
        img_end = img_resolution* 3 // 4 + 1
        #Disable all rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        # r_ai_cell = self.radius_target_a[stage][0]
        # is_neighbour = np.zeros(len(self.cell_name_interpolations_neighbour_a[stage][:,timestep]))
        # ai_location = self.ai.get_observation()
        # for m in range(len(self.cell_name_interpolations_neighbour_a[stage][:,timestep])):
        #     dist = np.linalg.norm(ai_location - self.pos_interpolations_neighbour_a[stage][m][timestep])
        #     r_cell = self.radius_neighbour_a[0][m]
        #     is_neighbour[m] = self.neighbor_model.predict([[dist, r_ai_cell, r_cell, len(self.pos_interpolations_a[stage][:,timestep])]])

        #Target cell
        p.resetBasePositionAndOrientation(agent[1],self.pos_interpolations_target_a[stage][1][timestep],orientation)
        p.resetBasePositionAndOrientation(agent[0],self.pos_interpolations_target_a[stage][0][timestep],orientation)

        for j in range(len(self.cell_name_interpolations_neighbour_a[stage])):
            #Observational state cell
            if self.cell_name_interpolations_neighbour_a[stage][j][timestep] in self.state_cell_list:
                p.resetBasePositionAndOrientation(cell[j],self.pos_interpolations_neighbour_a[stage][j][timestep],orientation) 

        # #Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        img_arr = p.getCameraImage(img_resolution,img_resolution)
        rgb = img_arr[2][img_start:img_end, img_start:img_end]
        depth = img_arr[3][img_start:img_end, img_start:img_end]
        depth_map = depth_conversion(depth)
        gray = rgb2gray(rgb)
        # plt.figure(1)
        # plt.title("64*64 Grey scale image")
        # image = plt.imshow(gray, cmap='Greys_r')
        # plt.figure(2)
        # plt.title("64*64 Depth image")
        # image = plt.imshow(depth,cmap='Greys_r')   # twilight_r
        # print("stage%d"%(stage*10+timestep))
        return [gray,depth_map]
    


if __name__ == '__main__':
    env = EmbryoBulletEnv('gui',embryo_num = 0)
    plt.ion()

    for i_episode in range(10):
        env.reset()
        counter = 0
        r_overall = 0
        while True:
            image = env.render()
            a = np.random.randint(8)
            s_, r, done, info = env.step(a)
            counter += 1
            r_overall += r
            if done:
                break
        print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',r_overall)
    
    env.close()
