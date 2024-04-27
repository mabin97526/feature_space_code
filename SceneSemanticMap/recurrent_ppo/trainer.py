import json

import cv2
import numpy as np
import os
import pickle
import torch
import time
from torch import optim
import tqdm
from tensoRF.network_cc import NeRFNetwork as CCNeRF
from buffer import Buffer
from model import ActorCriticModel
from nerf.provider import nerf_matrix_to_ngp
from nerf.utils import get_rays
from worker import Worker
from utils import create_env
from utils import polynomial_decay
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from normalization import RewardScaling
from yaml_parser import YamlParser
class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device
        self.run_id = run_id
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]
        self.continuous = config["continuous"]
        self.use_reward_scaling = config["use_reward_scaling"]
        self.exp_name = config["exp_name"]
        self.use_compose = config["use_compose"]
        self.global_episode = 0
        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp+self.exp_name)

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["environment"],worker_id=10)
        self.observation_space = dummy_env.observation_space
        self.action_space_shape = (dummy_env.action_space.shape[0],) if self.continuous else  (dummy_env.action_space.n,)
        dummy_env.close()
        if self.use_reward_scaling:
            self.reward_scaling = [RewardScaling(1, 0.99) for _ in range(self.config["n_workers"])]
        else:
            print("don't use reward scaling")
        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, self.observation_space, self.action_space_shape, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, self.observation_space, self.action_space_shape).to(self.device)
        self.model.train()
        print(self.model)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"],worker_id=w) for w in range(self.config["n_workers"])]

        # Setup observation placeholder
        if (config["environment"]["type"] == "UnityCarEnv" or config["environment"]["type"]=="UnityCarRace") and not config["use_compose"]:
            self.obs = np.zeros((self.config["n_workers"],) + self.observation_space[0].shape, dtype=np.float32)
        elif config["environment"]["type"] == "UnityCarEnv" and config["use_compose"]:
            self.obs = np.zeros((self.config["n_workers"],)+(84,84,3),dtype=np.float32)
        else:
            self.obs = np.zeros((self.config["n_workers"],) + self.observation_space.shape, dtype=np.float32)

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(self.config["n_workers"], self.device)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_cell = (hxs, cxs)

        if self.use_compose:
            print("Step 5 : Init compose model")
            self.nerf_config = YamlParser("./configs/CCNeRF.yaml").get_config()
            self.cube = self.load_model('./composemodel/cube_feature.pth')
            self.ball = self.load_model('./composemodel/ball_feature.pth')
            model = self.get_EmptyScene()
            model.eval()
            self.intrinsics = np.array([self.nerf_config["fl_x"],self.nerf_config["fl_y"],self.nerf_config["cx"],self.nerf_config["cy"]])
            self.transform_path = self.nerf_config["path"]
            with open(self.transform_path, 'r') as f:
                transform = json.load(f)
            frames = transform["frames"]
            poses = []
            for f in tqdm.tqdm(frames,desc=f'Loading test data'):
                pose = np.array(f['transform_matrix'],dtype=np.float32)
                pose = nerf_matrix_to_ngp(pose,scale = self.nerf_config["scale"],offset=self.nerf_config["offset"])
                poses.append(pose)
            poses = torch.from_numpy(np.stack(poses,axis=0)).to(device)
            radius = poses[:,:3,3].norm(dim=-1).mean(0).item()
            rays = get_rays(poses,self.intrinsics,self.nerf_config["H"],self.nerf_config["W"],-1)
            self.data = {
                'H' : self.nerf_config["H"],
                'W' : self.nerf_config["W"],
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
            }
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.nerf_config["fp16"]):
                    preds = self.test_step(self.data, model)
                s = preds[0].detach().cpu().numpy()
                s = (s * 255).astype(np.uint8)
            print("load success!")


        else:
            print("step 5 : Not use compose model")
        # Reset workers (i.e. environments)
        print("Step 6: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            obs = worker.child.recv()
            if self.use_compose:
                model = self.get_EmptyScene()
                self.process_state(state=obs,model=model,obj1=self.cube,obj2=self.ball)
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.nerf_config["fp16"]):
                        preds = self.test_step(self.data, model)
                    s = preds[0].detach().cpu().numpy()
                    s = (s * 255).astype(np.uint8)
                self.obs[w] = s
            else:
                self.obs[w] = obs
        if self.use_reward_scaling:
            for rs in self.reward_scaling:
                rs.reset()

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 7: Starting training")
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            # Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_info(episode_infos)

            # Print training statistics
            if "success_percent" in episode_result and "reward_mean" in episode_result:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success = {:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success_percent"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            else:
                '''result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))'''
                result = "{:4}  pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update,
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2],
                    torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            print(result)

            # Write training statistics to tensorboard
            self._write_training_summary(update, training_stats, episode_result)
            if(update%30==0):
                self._save_model(update)

            # Free memory
            del(self.buffer.samples_flat)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Save the trained model at the end of the training
        self._save_model(self.config["updates"])

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Save the initial observations and recurrentl cell states
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                elif self.recurrence["layer_type"] == "lstm":
                    self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0)
                    self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model to retrieve the policy, the states' value and the recurrent cell states
                #(1,84,84,3)
                policy, value, self.recurrent_cell = self.model(torch.tensor(self.obs), self.recurrent_cell, self.device)
                self.buffer.values[:, t] = value

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                if self.continuous:
                    action = policy.sample().detach()
                    #print(action)
                    for a in action.split(1,0):
                        actions.append(a)
                    action_logprob = policy.log_prob(action).detach()
                    for a_p in action_logprob.split(1,0):
                        log_probs.append(a_p)

                else:
                    for action_branch in policy:
                        action = action_branch.sample()
                        actions.append(action)
                        log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, rewards, self.buffer.dones[w, t], info = worker.child.recv()
                self.buffer.rewards[w, t] = self.reward_scaling[w](rewards) if self.use_reward_scaling else rewards
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    self.global_episode += 1
                    self.writer.add_scalar("epsidoe_reward_episodes",info['reward'],self.global_episode)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Reset recurrent cell states
                    if self.recurrence["reset_hidden_state"]:
                        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
                        if self.recurrence["layer_type"] == "gru":
                            self.recurrent_cell[:, w] = hxs
                        elif self.recurrence["layer_type"] == "lstm":
                            self.recurrent_cell[0][:, w] = hxs
                            self.recurrent_cell[1][:, w] = cxs
                # Store latest observations
                if self.use_compose:
                    model = self.get_EmptyScene()
                    self.process_state(obs,model,self.cube,self.ball)
                    model.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.nerf_config["fp16"]):
                            preds = self.test_step(self.data, model)
                        s = preds[0].detach().cpu().numpy()
                        s = (s * 255).astype(np.uint8)
                    cv2.imshow("state", s)
                    cv2.waitKey(1)
                    obs = s

                self.obs[w] = obs
                            
        # Calculate advantages
        _, last_value, _ = self.model(torch.tensor(self.obs), self.recurrent_cell, self.device)
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(self.config["epochs"]):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
        return train_info

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Forward model
        policy, value, _ = self.model(samples["obs"], recurrent_cell, self.device, self.buffer.actual_sequence_length)
        
        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        if self.continuous:
            log_probs = policy.log_prob(samples["actions"])
            entropies = policy.entropy()
        else:
            for i, policy_branch in enumerate(policy):
                log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
                entropies.append(policy_branch.entropy())
                log_probs = torch.stack(log_probs, dim=1)
                entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)
        
        # Remove paddings
        value = value[samples["loss_mask"]]
        log_probs = log_probs[samples["loss_mask"]]
        entropies = entropies[samples["loss_mask"]] 

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy()]

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("training/sequence_length", self.buffer.true_sequence_length, update)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)

    @staticmethod
    def _process_episode_info(episode_info:list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Arguments:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def _save_model(self,update_id=0) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id +str(update_id)+time.strftime("%Y%m%d-%H%M%S")+ ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id+"update"+str(update_id) +time.strftime("%Y%m%d-%H%M%S")+ ".nn")
        self.model.cuda()

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)

    def get_EmptyScene(self):
        model = CCNeRF(
            rank_vec_density=[1],
            rank_mat_density=[1],
            rank_vec=[1],
            rank_mat=[1],
            resolution=[1] * 3,  # fake resolution
            bound=self.nerf_config["bound"],  # a large bound is needed
            cuda_ray=self.nerf_config["cuda_ray"],
            density_scale=1,
            min_near=self.nerf_config["min_near"],
            density_thresh=self.nerf_config["density_thresh"],
            bg_radius=self.nerf_config["bg_radius"],
        ).to(self.device)
        return model

    def load_model(self,path):
        checkpoint_dict = torch.load(path, map_location=self.device)
        model = CCNeRF(
            rank_vec_density=checkpoint_dict['rank_vec_density'],
            rank_mat_density=checkpoint_dict['rank_mat_density'],
            rank_vec=checkpoint_dict['rank_vec'],
            rank_mat=checkpoint_dict['rank_mat'],
            resolution=checkpoint_dict['resolution'],
            bound=self.nerf_config["bound"],
            cuda_ray=self.nerf_config["cuda_ray"],
            density_scale=1,
            min_near=self.nerf_config["min_near"],
            density_thresh=self.nerf_config["density_thresh"],
            bg_radius=self.nerf_config["bg_radius"],
        ).to(self.device)
        model.load_state_dict(checkpoint_dict['model'], strict=False)
        return model

    def test_step(self,data, model, bg_color=None, perturb=False, device='cuda'):
        rays_o = data['rays_o']
        ## current:4096 *3 target:7225 * 3
        rays_d = data['rays_d']

        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(device)

        outputs = model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb)

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        return pred_rgb

    def process_state(self,state, model, obj1=None, obj2=None):
        s = np.array_split(state,3)
        for r in s:
            if ((r[0] == 0) and (r[1] == 1)):
                #print(r[2])
                model.compose(obj1, s=0.5, t=np.array([r[3]-1, r[6]/10, (r[2]-0.5)*2.2]))
            elif r[0] == 1 and r[1] == 0:
                #print(r[2])
                model.compose(obj2, s=1 , t=np.array([r[3]-1, r[6]/10,(r[2]-0.5)*2.2]))
            else:
                continue