import numpy as np
from .reward_function_base import BaseRewardFunction


class TimeoutReward(BaseRewardFunction):
    """
    TimeoutReward

    """
    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 500)



    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        new_reward = 0.
        if env.current_step >= self.max_steps:
            new_reward = 10.
            # print("Get the reward of timeout.\n")
        return self._process(new_reward, agent_id)
