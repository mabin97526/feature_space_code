

from typing import Callable
from stable_baselines3 import PPO

def linear_schedule(initial_value:float) -> Callable[[float],float]:
    def func(progress_remaining: float) ->float:
        return progress_remaining * initial_value

    return func



if __name__ == '__main__':
    model = PPO("MlpPolicy","CartPole-v1",learning_rate=linear_schedule(0.001))