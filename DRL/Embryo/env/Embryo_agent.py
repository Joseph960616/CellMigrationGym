import numpy as np
import pybullet as p
import time

FACTOR = 1/2**0.5
ACTION_LIST = (np.array([[1,0,0],[FACTOR,FACTOR,0],
                  [0,1,0],[-FACTOR,FACTOR,0],
                  [-1,0,0],[-FACTOR,-FACTOR,0],
                  [0,-1,0],[FACTOR,-FACTOR,0]]))

class EmbryoAgent():
    def __init__(self, client, agent, name):
        self.client = client
        self.agent = agent
        self.action_list = ACTION_LIST
        self.speed_base = 0.0
        self.name = name

    def get_ids(self):
        return self.agent, self.client

    def apply_action(self, action):
        """
        map action into agent movement parameters and apply to the agent
        """
        action = self.action_list[action]
        ########### add noise to ai moving speed##################
        speed = (self.speed_base + np.random.normal(0, self.speed_base  / 5, 1)[0]) * action
        p.resetBaseVelocity(self.agent,linearVelocity = speed, physicsClientId = self.client)
        # print(speed)
        
        return speed

    def get_observation(self):
        """
        Get the position of the agent in the simulation
        """
        pos, orientation = p.getBasePositionAndOrientation(self.agent, self.client)
        # Get the velocity of the cell
        # vel = p.getBaseVelocity(self.agent, self.client)[0]
        observation = [round(num,2) for num in pos]
        
        return observation

if __name__ == '__main__':
    p.connect(p.DIRECT)
    p.loadMJCF('./Embryo/utils/agent.xml')
    print("\nTesting...")
    agent = EmbryoAgent(0,0)
    ai_speed = agent.apply_action(1)
    obs = agent.get_observation()
    print(type(ai_speed),type(obs))
    print("ai_speed:{}".format(ai_speed))
    print("observation:{}".format(obs))
    print("\nProcess Completed\n")
    p.disconnect()
