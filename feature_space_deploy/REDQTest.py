import gymnasium as gym
from stable_baselines3 import REDQ

if __name__ == '__main__':
    env = gym.make("Pendulum-v1", render_mode = "human")
    policy_kwargs = dict(n_critics=4)
    model = REDQ("MlpPolicy",env,policy_kwargs=policy_kwargs,verbose=1)
    # model.learn(total_timesteps=100000,log_interval=1)
    # model.save("redq_pendulum")
    model.set_parameters(load_path_or_dict="./redq_pendulum.zip")
    s = env.reset()[0]
    done = False
    while not done:
        action, _states = model.predict(observation=s, deterministic=True)
        s, reward, done, truncated, info = env.step(action)
    env.close()


