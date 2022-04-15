from isaacgym import gymapi, gymutil, gymtorch
from stable_baselines3.common.vec_env import VecEnv
from scipy.spatial.transform import Rotation as R
import torchgeometry as tgm
import numpy as np
import torch
import torch.nn as nn
import time
import gym

class StateEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StateEstimator, self).__init__()

        self.model = nn.Sequential(
                                    nn.Linear(input_dim, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, output_dim)
                                )

    def forward(self, x):
        return self.model(x)

class HistoryBuffer:
    def __init__(self, ctx):
        self.ctx = ctx

        self.timers = torch.zeros(ctx.num_envs)
        self.prev_timers = torch.zeros(ctx.num_envs)

        self.dof_pos_err = torch.zeros(3, ctx.num_envs, ctx.num_dof).cuda()
        self.dof_vel = torch.zeros(3, ctx.num_envs, ctx.num_dof).cuda()

        self.target_dof_pos = torch.zeros(3, ctx.num_envs, ctx.num_dof).cuda()
        self.target_dof_pos[:, :] = ctx.default_dof_pos

    def update_history(self, tensor, indices, new_data):
        tensor[:2, indices, :] = tensor[1:, indices, :]
        tensor[0, indices, :] = new_data[indices, :]

    def step(self, cand_dof_pos_err, cand_dof_vel, cand_target_dof_pos):
        self.timers += self.ctx.dt

        update_idx = (self.timers - self.prev_timers) > 0.02

        self.update_history(self.dof_pos_err, update_idx, cand_dof_pos_err)
        self.update_history(self.dof_vel, update_idx, cand_dof_vel)
        self.update_history(self.target_dof_pos, update_idx, cand_target_dof_pos)
        self.prev_timers[update_idx] = self.timers[update_idx]

    def get_target_dof_pos(self):
        return self.target_dof_pos[:2, :, :]

    def get_dof_pos_err(self):
        return self.dof_pos_err

    def get_dof_vel(self):
        return self.dof_vel

    def reset(self, env_idx = None):
        if env_idx is None:
            self.dof_pos_err[:] = 0
            self.dof_vel[:] = 0
            self.target_dof_pos = self.ctx.default_dof_pos
            self.timers[:] = 0
            self.prev_timers[:] = 0
        else:
            self.dof_pos_err[:, env_idx, :] = 0
            self.dof_vel[:, env_idx, :] = 0

            self.target_dof_pos[:, env_idx, :] = self.ctx.default_dof_pos
            self.timers[env_idx] = 0
            self.prev_timers[env_idx] = 0

class SEObservation():
    def __init__(self, ctx, dyn):
        self.se = StateEstimator(input_dim = ctx.se_input_dim, output_dim = ctx.se_output_dim).cuda()
        self.se_loss = nn.MSELoss()
        self.se_optim = torch.optim.SGD(self.se.parameters(), lr = 0.0001, momentum = 0.9, nesterov = True)

        self.num_envs = ctx.num_envs
        self.history = HistoryBuffer(ctx)
        self.dyn = dyn

    def get_simulated_state(self):
        lin_vel = self.dyn.get_linear_velocity() # (num_envs, 3)
        feet_height = self.dyn.get_feet_position()[:, :, 2] # (num_envs, 4)
        contact_prob = self.dyn.get_feet_collsions().type(torch.LongTensor).cuda() # (num_envs, 4)

        res = torch.cat([lin_vel, feet_height, contact_prob], dim = 1)
        return res

    def get_se_observations(self):
        rot = self.dyn.get_rotation() # (num_envs, 4)
        ang_vel = self.dyn.get_angular_velocity() # (num_envs, 3)
        dof_pos = self.dyn.get_dof_position() # (num_envs, 12)
        dof_vel = self.dyn.get_dof_velocity() # (num_envs, 12)

        latest_dof_pos = self.history.get_target_dof_pos() # (2, num_envs, 12)
        t_dof_pos_t1 = latest_dof_pos[1, :, :] # (num_envs, 12)
        t_dof_pos_t2 = latest_dof_pos[0, :, :] # (num_envs, 12)

        hist_dof_pos = self.history.get_dof_pos_err().view(self.num_envs, -1) # (num_envs, 3 * 12)
        hist_dof_vel = self.history.get_dof_vel().view(self.num_envs, -1) # (num_envs, 3 * 12)

        rel_feet_pos = self.dyn.get_feet_position(relative = True).view(self.num_envs, -1) # (num_envs, 3 * 4)

        vel_cmd = torch.Tensor([[1.0, 0, 0] for _ in range(self.num_envs)]).cuda() # (num_envs, 3)

        obs_list = [rot, ang_vel, dof_pos, dof_vel, t_dof_pos_t1, t_dof_pos_t2, hist_dof_pos, hist_dof_vel, rel_feet_pos, vel_cmd]
        obs = torch.cat(obs_list, dim = 1) # (num_envs, 143)

        return obs

    def get(self, env_idx = None):
        se_obs = self.get_se_observations()

        self.se.train()
        predicted_state = self.se(se_obs)

        if env_idx is None:
            actual_state = self.get_simulated_state()

            loss = self.se_loss(predicted_state, actual_state)

            self.se_optim.zero_grad()
            loss.backward()
            self.se_optim.step()

            predicted_state = actual_state
            loss = loss.cpu().detach().numpy()
        else:
            se_obs = se_obs[env_idx, :]
            predicted_state = predicted_state[env_idx, :]
            loss = None

        full_obs = torch.cat([se_obs, predicted_state], dim = 1)
        if torch.isnan(full_obs).any():
            raise RuntimeError('NaN encountered in full_obs')

        return full_obs, loss