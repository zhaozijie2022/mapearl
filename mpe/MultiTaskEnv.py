# 用于试验多任务的环境, 每个环境有一个index, 使用这个index作为随机种子, 生成不同的任务
from mpe.environment import MultiAgentEnv


class MultiTaskEnv(MultiAgentEnv):
    def __init__(self, task_index=0, 
                 world=None, reset_callback=None, reward_callback=None, 
                 observation_callback=None, info_callback=None, 
                 done_callback=None, shared_viewer=True):
        super(MultiTaskEnv, self).__init__(
                world, reset_callback, reward_callback,
                observation_callback, info_callback,
                done_callback, shared_viewer)
        self.task_index = task_index








