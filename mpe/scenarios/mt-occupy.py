# scenario: occupy, 用于将pearl的单智能体多任务算法应用到mpe中
# 一个agent去追逐一个landmark   
# done: agent与landmark边界重合时done
# reward: -1 * dist (agent与landmark的距离)
# obs: 自身位置 (landmark的位置不作为obs)
# info: 无


import numpy as np
from mpe.core import World, Agent, Landmark
# 无需新定义world，直接使用mpe.core.World
from mpe.scenarios.BaseScenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, task_index=0):
        self.task_index = task_index

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 1
        num_landmarks = 1
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        original_state = np.random.get_state() # 保存随机种子
        np.random.seed(self.task_index) # 将种子设定为task_index
        pos_landmarks = np.random.uniform(-1, +1, (len(world.landmarks), world.dim_p))
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_pos = pos_landmarks[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
        np.random.set_state(original_state) # 恢复随机种子

    def reward(self, agent, world):
        return -np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)

    def observation(self, agent, world):
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos])
    
    def done(self, agent, world):
        return np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos) < 0.1
