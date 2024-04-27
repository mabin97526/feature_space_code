from stable_baselines3.common.env_checker import check_env
from UnityEnvClass.Unity_Base_env import *

if __name__ == '__main__':
    env = UnityBaseEnv("UnityEnv/CarSearchNoObsImgandRay",worker_id=0)
    print(env.observation_space)
    print(env.action_space)
    print(spaces.Space)
    print(isinstance(env.action_space, spaces.Space))
    check_env(env,True)
    #print(env.observation_space)
    info = env.reset()
    env.close()


    '''
    
    Box(0, 255, (96, 96, 3), uint8)
    Box([-1.  0.  0.], 1.0, (3,), float32)
    '''
    #