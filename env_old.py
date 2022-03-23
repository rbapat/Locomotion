from isaacgym import gymapi, gymutil, gymtorch
from scipy.spatial.transform import Rotation as R
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
import torch
import torch.nn as nn
import gym

import simulation as sm

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

class ConcurrentTrainingEnv(VecEnv):
    def __init__(self, num_envs, asset_path, asset_name, env_shape = (7, 7, 7)):

        self.action_space = gym.spaces.Box(low = np.zeros(12) - 1000000, high = np.zeros(12) + 1000000, dtype = np.float32)
        self.observation_space = gym.spaces.Box(low = np.zeros(153) - 1000000, high = np.zeros(153) + 1000000, dtype = np.float32)

        self.sim_helper = sm.SimHelper()
        self.num_envs = num_envs
        self.env_shape = env_shape

        asset_props = { 'driveMode': gymapi.DOF_MODE_POS, 'stiffness': 400, 'damping': 40, 'friction': 0.2 }

        self.gym, self.sim = self.sim_helper.get_gym()
        self.asset_handle, self.envs, self.actors = self.sim_helper.create_envs(asset_path, asset_name, num_envs, env_shape, asset_props)
        self.num_dof = self.gym.get_asset_dof_count(self.asset_handle)

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        self.default_dof_pos = torch.Tensor([[0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6] for _ in range(num_envs)])
        self.default_dof_vel = torch.zeros_like(self.default_dof_pos)

        self.history = sm.HistoryBuffer(self.num_envs, self.default_dof_pos[0], self.default_dof_vel[0])
        self.dyn = sm.RobotDynamics(self.gym, self.sim)

        self.se = StateEstimator(input_dim = 142, output_dim = 11).cuda()
        self.se_loss = nn.MSELoss()
        self.se_optim = torch.optim.SGD(self.se.parameters(), lr = 0.001, momentum = 0.9, nesterov = True)

    def reset(self, env_idx = None):
        if env_idx is None:
            random_pos = sm.sample_data((-1, -1, 0.36), self.env_shape, num_times = self.num_envs)
            random_rot = R.from_euler('xyz', sm.sample_data((0, 0, -1), (360, 360, 360), num_times = self.num_envs), degrees = True)

            self.dyn.set_position(random_pos)
            self.dyn.set_rotation(random_rot)
            self.dyn.set_dof_position(self.default_dof_pos)
            self.dyn.set_dof_velocity(self.default_dof_vel)

            self.history.reset(env_idx)

            self.dyn.step(step_physics = False)
            return self.run_state_estimation().numpy()
        else:
            random_pos = sm.sample_data((-1, -1, 0.36), self.env_shape)
            random_rot = R.from_euler('xyz', sm.sample_data((0, 0, -1), (360, 360, 360)), degrees = True)

            self.dyn.set_position(random_pos, env_idx = env_idx)
            self.dyn.set_rotation(random_rot, env_idx = env_idx)
            self.dyn.set_dof_position(self.default_dof_pos[env_idx], env_idx = env_idx)
            self.dyn.set_dof_velocity(self.default_dof_vel[env_idx], env_idx = env_idx)


            self.history.reset(env_idx = env_idx)
            self.dyn.step(step_physics = False)

            return self.run_state_estimation(env_idx).numpy()
    
    def get_observations(self, env_idx = None):
        rot = torch.Tensor(self.dyn.get_rotation().as_quat()) # (num_envs, 4)
        ang_vel = self.dyn.get_angular_velocity() # (num_envs, 3)
        dof_pos = self.dyn.get_dof_position() # (num_envs, 12)
        dof_vel = self.dyn.get_dof_velocity() # (num_envs, 12)

        latest_dof_pos = self.history.get_dof_pos() # (2, num_envs, 12)
        t_dof_pos_t1 = latest_dof_pos[1, :, :] # (num_envs, 12)
        t_dof_pos_t2 = latest_dof_pos[0, :, :] # (num_envs, 12)

        hist_dof_pos = self.history.get_dof_pos_history().view(self.num_envs, -1) # (num_envs, 3 * 12)
        hist_dof_vel = self.history.get_dof_vel_history().view(self.num_envs, -1) # (num_envs, 3 * 12)

        rel_feet_pos = self.dyn.get_feet_position(relative = True).view(self.num_envs, -1) # (num_envs, 3 * 4)

        vel_cmd = torch.Tensor([[-1, 0, 0] for _ in range(self.num_envs)])# (num_envs, 3)

        obs_list = [rot, ang_vel, dof_pos, dof_vel, t_dof_pos_t1, t_dof_pos_t2, hist_dof_pos, hist_dof_vel, rel_feet_pos, vel_cmd]
        obs = torch.cat(obs_list, dim = 1) # (num_envs, 143)

        if env_idx is None:
            return obs
        else:
            return obs[env_idx, :]

    def get_simulated_state(self):
        lin_vel = self.dyn.get_linear_velocity() # (num_envs, 3)
        feet_height = self.dyn.get_feet_position()[:, 2, :] # (num_envs, 4)
        contact_prob = self.dyn.get_feet_collsions()[1].type(torch.LongTensor) # (num_envs, 4)

        return torch.cat([lin_vel, feet_height, contact_prob], dim = 1)

    def run_state_estimation(self, env_idx = None):
        # normalize obs
        obs = self.get_observations(env_idx)
        self.se.train()
        predicted_state = self.se(obs.cuda())
        if env_idx is None:
            
            
            actual_state = self.get_simulated_state().cuda()

            loss = self.se_loss(predicted_state, actual_state)

            self.se_optim.zero_grad()
            loss.backward()
            self.se_optim.step()
        else:
            obs = obs.unsqueeze(0)
            predicted_state = predicted_state.unsqueeze(0)

        return torch.cat([obs, predicted_state.cpu()], dim = 1).detach()

    def compute_rewards(self, obs, target_dof_pos, k_v = 3, k_w = 3, k_a = 3, k_slip = -0.08, k_cl = -15, k_ori = -3, k_t = -6e-4, k_q = -0.75, k_qdot = -6e-4, k_qddot = -0.02, k_s1 = -2.5, k_s2 = -1.2, k_base = -1.5):
        rot = R.from_quat(obs[:, :4])
        ang_vel = obs[:, 4:7]
        dof_pos = obs[:, 7:19]
        dof_vel = obs[:, 19:31]
        t_dof_pos_t1 = obs[:, 31:43]
        t_dof_pos_t2 = obs[:, 43:55]
        hist_dof_pos = obs[:, 55:91].view(self.num_envs, 3, 12)
        hist_dof_vel = obs[:, 91:127].view(self.num_envs, 3, 12)
        rel_feet_pos = obs[:, 127:139].view(self.num_envs, 3, 4)
        vel_cmd = obs[:, 139:143]
        se_lin_vel = obs[:, 143:146]
        se_feet_height = obs[:, 146:150]
        se_contact_prob = obs[:, 150:154]

        cur_feet_pos = self.dyn.get_feet_position() # (num_envs, 3, 4)
        cur_feet_vel = self.dyn.get_feet_velocity() # (num_envs, 3, 4)
        contact_prob = self.dyn.get_feet_collsions()[1].type(torch.FloatTensor) # (num_envs, 4)
        cur_lin_vel = self.dyn.get_linear_velocity()
        cur_ang_vel = self.dyn.get_angular_velocity()

        vel_err = vel_cmd[:, :2] - cur_lin_vel[:, :2]
        r_v = k_v * torch.exp(-1 * torch.diagonal(vel_err @ vel_err.T))
        r_w = k_w * torch.exp(-1.5 * (vel_cmd[:, 2] - cur_ang_vel[:, 2])**2)
        r_air = torch.zeros(self.num_envs) # TODO

        f_vels_sq = torch.norm(cur_feet_vel, dim = 1)
        f_vels = torch.square(f_vels_sq)

        r_slip = torch.diagonal((k_slip * contact_prob[:, :3]) @ f_vels[:, :3].T)

        des_feet_h = 0.09
        delta_h = k_cl * torch.square(cur_feet_pos[:, 2, :] - des_feet_h)
        r_cl = torch.diagonal(delta_h @ f_vels_sq.T)
        r_ori = torch.zeros(self.num_envs) # k_ori * torch.Tensor(rot.as_euler('xyz')[:, 2])
        r_t = torch.zeros(self.num_envs) # TODO

        dof_pos_error = dof_pos - self.default_dof_pos
        r_q = k_q * torch.diagonal(dof_pos_error @ dof_pos_error.T)
        r_qdot = k_qdot * torch.diagonal(dof_vel @ dof_vel.T)

        dof_vel_error = dof_vel - hist_dof_vel[:, 0, :]
        r_qddot = k_qddot * torch.diagonal(dof_vel_error @ dof_vel_error.T)

        t_dof_error = target_dof_pos - t_dof_pos_t1
        r_s1 = k_s1 * torch.diagonal(t_dof_error @ t_dof_error.T)

        t_dof_error = t_dof_error - t_dof_pos_t1 + t_dof_pos_t2
        r_s2 = k_s2 * torch.diagonal(t_dof_error @ t_dof_error.T)

        r_base = k_base * (0.8 * cur_lin_vel[:, 2] + 0.2 * torch.abs(cur_ang_vel[:, 0]) + 0.2 * cur_ang_vel[:, 1])
        

        pos_reward = r_v + r_w + r_air
        neg_reward = r_slip + r_cl + r_ori + r_t + r_q + r_qdot + r_qddot + r_s1 + r_s2 + r_base

        total_reward = pos_reward * torch.exp(0.2 * neg_reward)
        return total_reward

    def step(self, actions):
        target_dof_pos = self.default_dof_pos + 0.5 * actions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_dof_pos))
        self.dyn.step()

        observations = self.run_state_estimation()
        rewards = self.compute_rewards(observations, target_dof_pos)        
        self.history.step(target_dof_pos, self.dyn.get_dof_position(), self.dyn.get_dof_velocity())

        observations = observations.numpy()
        dones = self.dyn.get_body_collisions()[0]

        for term_idx in torch.where(dones)[0]:
            rewards[term_idx] = -10
            observations[term_idx, :] = self.reset(term_idx)

        infos = [{}] * self.num_envs

        return observations, rewards.numpy(), dones.numpy(), infos

    def render(self):
        self.gym.step_graphics(self.sim);
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def close(self):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices = None):
        raise NotImplementedError

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