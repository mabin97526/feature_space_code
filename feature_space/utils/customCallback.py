from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    def __init__(self,verbose = 0,record_intention=False):
        super(TensorboardCallback,self).__init__(verbose)
        self.interval = 0
        self.episode_num = 0
        self.record_intention = record_intention
        if self.record_intention:
            self.predict_ = "./temp/predict_stat"
            self.truth_ = "./temp/truth_stat"
    def _on_step(self) -> bool:
        if(self.interval%10==0):
            action_0 = self.locals['actions'][0]
            self.logger.record("action_0",action_0)
            self.logger.dump(self.num_timesteps)
            self.interval = 0;
        self.interval += 1;
        for idx,done in enumerate(self.locals['dones']):
            if done:
                e_length = self.locals['infos'][idx]['length']
                self.logger.record("episode_reward",self.locals['infos'][idx]['reward'])
                self.logger.record("episode_length",self.locals['infos'][idx]['length'])
                if self.record_intention:
                    #behaviour_intention_acc = self.model.behaviour_intention_acc/e_length
                    #self.model.behaviour_intention_acc = 0
                    action_intention_acc = self.model.action_intention_acc/e_length
                    self.model.action_intention_acc = 0
                    #self.logger.record("behaviour_intention_acc",behaviour_intention_acc)
                    self.logger.record("action_intention_acc",action_intention_acc)
                    pred = self.predict_ + str(self.num_timesteps) + ".npz"
                    tru = self.truth_ + str(self.num_timesteps) + ".npz"
                    obs_1 = np.savez(pred, array1=self.model.obs_predic)
                    obs_2 = np.savez(tru,array1=self.model.obs_)
                    self.model.obs_predic = []
                    self.model.obs_ = []
                self.logger.dump(self.num_timesteps)





