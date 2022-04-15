from isaacgym import gymapi, gymutil, gymtorch
from stable_baselines3.common.vec_env import VecEnv
from scipy.spatial.transform import Rotation as R
import torchgeometry as tgm
import numpy as np
import torch
import torch.nn as nn
import time
import gym

from simulation import SimHelper
from agent import QuadrupedAgent

class Context:

    default_dof_pos = torch.Tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).cuda()
    action_scale = 0.1
    dt = 1. / 60.
    max_ep_length = 20

    max_ep_steps = max_ep_length / dt

    se_input_dim = 142
    se_output_dim = 11

    def __init__(self, sim, gym, num_envs, asset_handle, env_shape):
        self.sim = sim
        self.gym = gym
        self.num_envs = num_envs
        self.asset_handle = asset_handle
        self.num_dof = self.gym.get_asset_dof_count(self.asset_handle)
        self.num_rb = self.gym.get_asset_rigid_body_count(self.asset_handle)
        self.env_shape = env_shape

class ConcurrentTrainingEnv(VecEnv):
    def __init__(self, asset_path, asset_name, num_envs, env_shape):
        self.sim_helper = SimHelper()
        self.all_envs = [i for i in range(num_envs)]

        # create simulation environments
        isaac_gym, sim = self.sim_helper.get_gym()
        asset_handle, envs, actors = self.sim_helper.create_envs(asset_path, asset_name, num_envs, env_shape)
        isaac_gym.prepare_sim(sim)

         # create context
        self.ctx = Context(sim, isaac_gym, num_envs, asset_handle, env_shape)

        # create visualization
        self.viewer = isaac_gym.create_viewer(sim, gymapi.CameraProperties())
        self.should_render = True
        isaac_gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_render")

        # create agent to train
        self.agent = QuadrupedAgent(self.ctx)

        # OpenAI Gym Environment required fields
        obs_space = self.ctx.se_input_dim + self.ctx.se_output_dim
        self.num_envs = self.ctx.num_envs
        self.observation_space = gym.spaces.Box(low = np.zeros(obs_space) - 1000000, high = np.zeros(obs_space) + 1000000, dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.zeros(12) - 1000000, high = np.zeros(12) + 1000000, dtype = np.float32)
        self.metadata = {"render_modes": []}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

    def reset(self, env_idx = None):
        if env_idx is None:
            env_idx = self.all_envs

        obs = self.agent.reset(env_idx).cpu().detach().numpy()
        return np.clip(obs, -10, 10)

    def step(self, actions):
        target_dof_pos = self.agent.act(actions)
        self.ctx.gym.set_dof_position_target_tensor(self.ctx.sim, gymtorch.unwrap_tensor(target_dof_pos))

        self.ctx.gym.simulate(self.ctx.sim)
        self.ctx.gym.fetch_results(self.ctx.sim, True)

        obs, rewards, dones, infos = self.agent.step_simulation(target_dof_pos)

        obs = obs.cpu().detach().numpy()
        rewards = rewards.cpu().detach().numpy()
        dones = dones.cpu().detach().numpy()

        return np.clip(obs, -10, 10), rewards, dones, infos

    def render(self):
        for event in self.ctx.gym.query_viewer_action_events(self.viewer):
            if event.action == 'toggle_render' and event.value > 0:
                self.should_render = not self.should_render

        if self.should_render:
            self.ctx.gym.step_graphics(self.ctx.sim);
            self.ctx.gym.draw_viewer(self.viewer, self.ctx.sim, True)
            #self.ctx.gym.sync_frame_time(self.ctx.sim)
        else:
            self.ctx.gym.poll_viewer_events(self.viewer)
    def close(self):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices = None):
        return [False]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def seed(self, seed = None):
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def step_async(self, actions):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError