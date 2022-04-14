from isaacgym import gymapi, gymutil, gymtorch
from scipy.spatial.transform import Rotation as R
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
import torch
import torch.nn as nn
import time
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

class K:
    default_dof_pos = torch.Tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6])
    default_dof_vel = torch.zeros_like(default_dof_pos)

    se_input_dim = 142
    se_output_dim = 11

    sigma_t = 0.1
    desired_feet_height = 0.09

    k_v = 3
    k_w = 3
    k_a = 0.3
    k_slip = -0.08
    k_cl = -15
    k_ori = -1
    k_t = -6e-4
    k_q = -0.75
    k_qdot = -6e-4
    k_qddot = -0.02
    k_s1 = -2.5
    k_s2 = -1.2
    k_base = -1.5
    

# TODO: allocate buffer to hold all observations: size (num_envs, output_dim)
# TODO: individual observations will be views of the buffer
class Observation():
    def __init__(self, dyn, num_envs):
        self.dyn = dyn
        self.num_envs = num_envs
        self.output_dim = K.se_input_dim + 11

        self.history = sm.HistoryBuffer(num_envs, K.default_dof_pos, K.default_dof_vel)

        self.se = StateEstimator(input_dim = K.se_input_dim, output_dim = K.se_output_dim).cuda()
        self.se_loss = nn.MSELoss()
        self.se_optim = torch.optim.SGD(self.se.parameters(), lr = 0.0001, momentum = 0.9, nesterov = True)

    def get_raw_observations(self, env_idx):
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

        vel_cmd = torch.Tensor([[1.0, 0, 0] for _ in range(self.num_envs)])# (num_envs, 3)

        obs_list = [rot, ang_vel, dof_pos, dof_vel, t_dof_pos_t1, t_dof_pos_t2, hist_dof_pos, hist_dof_vel, rel_feet_pos, vel_cmd]
        obs = torch.cat(obs_list, dim = 1) # (num_envs, 143)

        if env_idx is None:
            return obs
        else:
            return obs[env_idx, :].unsqueeze(0)

    def get_simulated_state(self):
        lin_vel = self.dyn.get_linear_velocity() # (num_envs, 3)
        feet_height = self.dyn.get_feet_position()[:, 2, :] # (num_envs, 4)
        contact_prob = self.dyn.get_feet_collsions()[1].type(torch.LongTensor) # (num_envs, 4)

        res = torch.cat([lin_vel, feet_height, contact_prob], dim = 1)
        # mean, std = torch.mean(res, dim = 0), torch.std(res, dim = 0) + 1e-8
        norm_res = res # torch.clip((res - mean) / std, -10, 10)

        return norm_res

    def get_observation(self, env_idx = None):
        # TODO: normalize obs

        obs = self.get_raw_observations(env_idx)

        if env_idx is None:
            mean, std = torch.mean(obs, dim = 0), torch.std(obs, dim = 0) + 1e-8
            norm_obs = obs # torch.clip((obs - mean) / std, -10, 10)
            loss = None
        else:
            mean, std = torch.mean(obs, dim = 0), torch.std(obs, dim = 0, unbiased=False) + 1e-8
            norm_obs = obs # torch.clip((obs - mean) / std, -10, 10)
            loss = None

        self.se.train()
        predicted_state = self.se(norm_obs.cuda())
        if env_idx is None:    
            actual_state = self.get_simulated_state().cuda()

            loss = self.se_loss(predicted_state, actual_state)

            self.se_optim.zero_grad()
            loss.backward()
            self.se_optim.step()

            loss = loss.cpu().detach().numpy()
            predicted_state = actual_state

        full_obs = torch.cat([norm_obs, predicted_state.cpu()], dim = 1).detach()
        if torch.isnan(norm_obs).any():
            print("ERROR!!!!")
            print(torch.isnan(std).any())
            input()
        return obs, full_obs, loss

class ConcurrentTrainingEnv(VecEnv):
    def __init__(self, num_envs, asset_path, asset_name, env_shape = (2, 2, 2)):

        self.sim_helper = sm.SimHelper()
        self.num_envs = num_envs
        self.env_shape = env_shape

        self.gym, self.sim = self.sim_helper.get_gym()
        self.asset_handle, self.envs, self.actors = self.sim_helper.create_envs(asset_path, asset_name, num_envs, env_shape)
        self.num_dof = self.gym.get_asset_dof_count(self.asset_handle)

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.should_render = True
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_render")

        self.dyn = sm.RobotDynamics(self.gym, self.sim)
        self.obs = Observation(self.dyn, self.num_envs)

        self.default_dof_pos = K.default_dof_pos.expand(self.num_envs, self.num_dof)
        self.default_dof_vel = K.default_dof_vel.expand(self.num_envs, self.num_dof)
        self.step_timer = time.perf_counter()

        self.contact_state = torch.zeros((self.num_envs, 4))
        self.takeoff_time = torch.zeros((self.num_envs, 4))
        self.touchdown_time = torch.zeros((self.num_envs, 4))
        self.ep_lens = torch.zeros((self.num_envs))

        obs_space = self.obs.output_dim
        self.observation_space = gym.spaces.Box(low = np.zeros(obs_space) - 1000000, high = np.zeros(obs_space) + 1000000, dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.zeros(12) - 1000000, high = np.zeros(12) + 1000000, dtype = np.float32)
        self.metadata = {"render_modes": []}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None
        

    def sample_data(self, orig, bounds, num_times):
        if num_times < 1:
            raise RuntimeError("Number of times to sample data should be > 0")
        else:
            out = np.zeros((num_times, len(orig)))
            mask = np.array(orig) == -1
            
            for i in range(num_times):
                random_arr = np.random.sample(len(bounds)) * bounds
                out[i, :] = (mask * random_arr) + (1 - mask) * orig

            return torch.from_numpy(out)

    def reset(self, env_idx = None):
        if env_idx is not None:
            self.takeoff_time[env_idx, :] = 0
            self.touchdown_time[env_idx, :] = 0
            self.ep_lens[env_idx] = 0
            num_times = 1
        else:
            self.takeoff_time[:] = 0
            self.touchdown_time[:] = 0
            self.ep_lens[:] = 0
            num_times = self.num_envs

        random_pos = self.sample_data((0, 0, 0.28), self.env_shape, num_times).squeeze()
        random_rot = R.from_euler('xyz', self.sample_data((0, 0, -1), (360, 360, 360), num_times).squeeze(), degrees = True)

        self.dyn.set_position(random_pos, env_idx = env_idx)
        self.dyn.set_linear_velocity(torch.zeros_like(random_pos), env_idx = env_idx)
        self.dyn.set_angular_velocity(torch.zeros_like(random_pos), env_idx = env_idx)
        self.dyn.set_rotation(random_rot, env_idx = env_idx)
        self.dyn.set_dof_position(self.default_dof_pos[env_idx], env_idx = env_idx)
        self.dyn.set_dof_velocity(self.default_dof_vel[env_idx], env_idx = env_idx)

        self.obs.history.reset(env_idx = env_idx)
        self.dyn.step(step_physics = False)

        return self.obs.get_observation(env_idx)[1].numpy()

    def compute_rewards(self, obs, target_dof_pos, delta):
        mag = lambda mat, dim = -1: torch.sum(mat * mat, dim = dim)

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

        cur_feet_pos = self.dyn.get_feet_position() # (num_envs, 3, 4)
        cur_feet_vel = self.dyn.get_feet_velocity() # (num_envs, 3, 4)
        contact_prob = self.contact_state.type(torch.FloatTensor) # (num_envs, 4)
        cur_lin_vel = self.dyn.get_linear_velocity()
        cur_ang_vel = self.dyn.get_angular_velocity()
        cur_joint_torque = self.dyn.get_joint_forces()

        '''
        #tracking_lin_vel = 1.0
        r_tracking_lin_vel = torch.exp(-1 * mag(vel_cmd[:, :2] - cur_lin_vel[:, :2]))

        #tracking_ang_vel = 0.5
        r_tracking_ang_vel = 0.5 * torch.exp(-1 * mag(vel_cmd[:, 2:] - cur_ang_vel[:, 2:]))

        #lin_vel_z = -2.0
        r_lin_vel_z = -2 * mag(cur_lin_vel[:, 2:])

        #ang_vel_xy = -0.05
        r_ang_vel_xy = -0.05 * mag(cur_ang_vel[:, :2])

        #orientation = -5
        pass

        #torques = -0.000025
        r_torques = -0.000025 * mag(cur_joint_torque)

        #dof_acc = -2.5e-7
        r_dof_acc = -2.5e-7 * mag(dof_vel - hist_dof_vel[:, 0, :]) / 0.2

        #feet_air_time =  2.0
        r_feet_air_time = 2 * torch.sum(self.takeoff_time - 0.1, dim = -1)

        #collision = -1.
        r_contact_force = -0.01 * torch.sum(self.dyn.get_feet_collsions(350)[1], dim = -1)

        #action_rate = -0.01
        r_action_rate = -0.01 * mag(target_dof_pos - t_dof_pos_t1)

        total_reward = r_tracking_lin_vel + r_tracking_ang_vel + r_lin_vel_z + r_ang_vel_xy + r_torques + r_dof_acc + r_feet_air_time + r_action_rate + r_contact_force
        total_reward[total_reward < 0] = 0

        terms = torch.mean(torch.stack([r_tracking_lin_vel, r_tracking_ang_vel, r_lin_vel_z, r_ang_vel_xy, r_torques, r_dof_acc, r_feet_air_time, r_action_rate, r_contact_force, total_reward]), dim = 1)
        names = ['r_tracking_lin_vel', 'r_tracking_ang_vel', 'r_lin_vel_z', 'r_ang_vel_xy', 'r_torques', 'r_dof_acc', 'r_feet_air_time', 'r_action_rate', 'r_contact_force', 'r_total']

        return total_reward, terms, names
        '''

        r_v = K.k_v * torch.exp(-1 * mag(vel_cmd[:, :2] - cur_lin_vel[:, :2]))
        r_w = torch.zeros(self.num_envs) # K.k_w * torch.exp(-1.5 * mag(vel_cmd[:, 2:] - cur_ang_vel[:, 2:]))

        maxes = torch.where(self.takeoff_time > self.touchdown_time, self.takeoff_time, self.touchdown_time)
        r_air = K.k_a * torch.where(maxes > 0.25, 0., maxes.type(torch.DoubleTensor))
        r_air = torch.sum(r_air, dim = -1)


        norm_feet_vel = mag(cur_feet_vel[:, :2, :].transpose(1, 2))
        r_slip = K.k_slip * norm_feet_vel
        r_slip[~self.contact_state] = 0
        r_slip = torch.sum(r_slip, dim = -1)

        delta_cl = mag(cur_feet_pos[:, 2:, :] - K.desired_feet_height, dim = 1)
        feet_vel_quart = norm_feet_vel**0.25
        r_cl = K.k_cl * delta_cl * feet_vel_quart
        r_cl = torch.sum(r_cl, dim = -1)

        rot_error = np.pi - torch.Tensor(rot.as_euler('xyz', degrees = False)[:, 2])
        r_ori = K.k_ori * torch.abs(rot_error)
        r_t = torch.zeros(self.num_envs) # K.k_t * mag(cur_joint_torque)
        r_q = K.k_q * mag(dof_pos - self.default_dof_pos)
        r_qdot = K.k_qdot * mag(dof_vel)
        r_qddot = K.k_qddot * mag(dof_vel - hist_dof_vel[:, 0, :])
        r_s1 = K.k_s1 * mag(target_dof_pos - t_dof_pos_t1)
        r_s2 = K.k_s2 * mag(target_dof_pos - 2 * t_dof_pos_t1 + t_dof_pos_t2)

        #r_base = K.k_base * (0.8 * cur_lin_vel[:, 2] + 0.2 * torch.abs(cur_ang_vel[:, 0]) + 0.2 * cur_ang_vel[:, 1])
        r_base = K.k_base * (0.2 * torch.abs(cur_ang_vel[:, 0]) + 0.2 * torch.abs(cur_ang_vel[:, 1]))

        pos_reward = r_v + r_w + r_air
        neg_reward = r_cl + r_ori + r_t + r_qddot #+ r_s1 + r_s2 + r_base # r_q + r_qdot + r_slip

        total_reward = pos_reward * torch.exp(0.2 * neg_reward)

        terms = torch.mean(torch.stack([total_reward, pos_reward, neg_reward, r_v, r_w, r_air, r_slip, r_cl, r_t, r_q, r_qdot, r_qddot, r_s1, r_s2, r_base, r_ori]), dim = 1)
        names = ['total', 'pos', 'neg', 'r_v', 'r_w', 'r_air', 'r_slip', 'r_cl', 'r_t', 'r_q', 'r_qdot', 'r_qddot', 'r_s1', 'r_s2', 'r_base', 'r_ori']

        return total_reward, terms, names


    def update_contact_states(self):
        new_time = time.perf_counter()
        self.contact_state = self.dyn.get_feet_collsions()[1]

        self.takeoff_time[self.contact_state] += 1. / 60. # (new_time - self.step_timer)
        self.takeoff_time[~self.contact_state] = 0

        self.touchdown_time[self.contact_state] = 0
        self.touchdown_time[~self.contact_state] += 1. / 60. # (new_time - self.step_timer)

        self.step_timer = new_time        

    def step(self, actions):
        target_dof_pos = self.default_dof_pos + K.sigma_t * actions

        prev_pos = self.dyn.get_position()

        goal_pos = prev_pos.clone()

        zrot = self.dyn.get_rotation().as_euler('xyz', degrees = True)[:, 2]
        goal_pos[:, 0] += 1 * np.cos(-1 * zrot)
        goal_pos[:, 1] += 1 * np.sin(-1 * zrot)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_dof_pos))
        self.dyn.step()

        self.update_contact_states()
        delta = (self.dyn.get_position() - goal_pos)[:, :2]

        observations, norm_obs, loss = self.obs.get_observation()
        rewards, terms, names = self.compute_rewards(observations, target_dof_pos, delta)        
        self.obs.history.step(target_dof_pos, self.dyn.get_dof_position(), self.dyn.get_dof_velocity())

        self.ep_lens += 1
        norm_obs = norm_obs.numpy()

        dones = self.dyn.get_body_collisions()[0]
        ep_dones = self.ep_lens > 20 * 60
        
        for term_idx in torch.where(ep_dones)[0]:
            norm_obs[term_idx, :] = self.reset(term_idx)

        for term_idx in torch.where(dones)[0]:
            rewards[term_idx] = -10
            norm_obs[term_idx, :] = self.reset(term_idx)

        infos = [{} for _ in range(self.num_envs)]
        infos[0]['terms'] = terms
        infos[0]['names'] = names
        infos[0]['loss'] = loss.item()

        return np.clip(norm_obs, -10, 10), rewards.numpy(), dones.numpy(), infos

    def render(self):
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == 'toggle_render' and event.value > 0:
                self.should_render = not self.should_render

        if self.should_render:
            self.gym.step_graphics(self.sim);
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

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