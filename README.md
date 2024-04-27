# feature_space_code
This project includes shared feature semantic space algorithms for sim-to-real transfer.It mainly contains five modules, including:feature space in unity, feature space in ros space, feature space deploy package, scene semantic map module and feature space based RL algorithms.
<!-- ## Features

- RL Baselines (mainly based on [StableBaselines]( https://github.com/DLR-RM/stable-baselines3))
- feature space deploy package(mainly based on [Yolov5](https://github.com/TomMao23/multiyolov5))
- Unity feature space([ML-Agents](https://github.com/Unity-Technologies/ml-agents))
- support visual camera, lidar, infrared camera
- Ros package in limo car
- Scene Semantic Map(mainly based on [torch-ngp](https://github.com/ashawkey/torch-ngp))
- [BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)
- [BarloTwins](https://arxiv.org/abs/2103.03230)
- [VICREG](https://arxiv.org/abs/2105.04906)
- VICREG_AP
- Siamese PPO
- Intention PPO
- recurrent PPO
- custom feature network
- feature extractor pretrain
- Tensorfusion 
- LowRankTensorFusion
- MULT
- Multi-Modal VAE


## Supported Env

Gym, PyBullet and Unity env with ML-Agents

Observation can be any combination of vectors and images, which means an agent can have multiple sensors and the resolution of each image can be different.

Action space can be both continuous and discrete.

## Example

### Train RL
In the feature_space/Experiment
```python
import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
```
- Pretrain feature extractor

```
cd utils
python feature_extractor_trainer.py
``` -->
