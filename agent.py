from isaacgym import gymapi, gymutil, gymtorch
from stable_baselines3.common.vec_env import VecEnv
from scipy.spatial.transform import Rotation as R
import torchgeometry as tgm
import numpy as np
import torch
import torch.nn as nn
import time
import gym

from dynamics import AgentDynamics
from estimator import SEObservation

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

class QuadrupedAgent:
    def __init__(self, ctx):
        self.ctx = ctx

        self.default_dof_pos = ctx.default_dof_pos.expand(ctx.num_envs, ctx.num_dof).cuda()
        self.default_dof_vel = torch.zeros_like(self.default_dof_pos).cuda()

        self.dyn = AgentDynamics(ctx)
        self.obs = SEObservation(ctx, self.dyn)

        self.takeoff_time = torch.zeros((ctx.num_envs, 4))
        self.touchdown_time = torch.zeros((ctx.num_envs, 4))
        self.ep_lens = torch.zeros((ctx.num_envs))

    def act(self, actions):
        actions = torch.from_numpy(actions).float().cuda()

        target_dof_pos = self.default_dof_pos + actions * self.ctx.action_scale
        return target_dof_pos

    def sample_data(self, orig, bounds, num_times):
        if num_times < 1:
            raise RuntimeError("Number of times to sample data should be > 0")
        else:
            out = np.zeros((num_times, len(orig)))
            mask = np.array(orig) == -1
            
            for i in range(num_times):
                random_arr = np.random.sample(len(bounds)) * bounds
                out[i, :] = (mask * random_arr) + (1 - mask) * orig

            return out

    def reset(self, env_idx):
        if env_idx is not None:
            self.takeoff_time[env_idx, :] = 0
            self.touchdown_time[env_idx, :] = 0
            self.ep_lens[env_idx] = 0
            num_times = len(env_idx)
        else:
            self.takeoff_time[:] = 0
            self.touchdown_time[:] = 0
            self.ep_lens[:] = 0
            num_times = self.ctx.num_envs


        random_pos = torch.from_numpy(self.sample_data((0, 0, 0.28), self.ctx.env_shape, num_times)).squeeze().float().cuda()
        random_rot = torch.from_numpy(R.from_euler('xyz', self.sample_data((0, 0, -1), (360, 360, 360), num_times)).as_quat()).squeeze().float().cuda()

        self.dyn.root_position[env_idx, :] = random_pos 
        self.dyn.root_lin_vel[env_idx, :] = torch.zeros_like(random_pos) 
        self.dyn.root_ang_vel[env_idx, :] = torch.zeros_like(random_pos) 
        self.dyn.root_rotation[env_idx, :] = random_rot
        self.dyn.dof_pos[env_idx, :] = self.default_dof_pos[env_idx]
        self.dyn.dof_vel[env_idx, :] = self.default_dof_vel[env_idx]
            
        stupid_reference = torch.cuda.IntTensor(env_idx)
        indices = gymtorch.unwrap_tensor(stupid_reference)

        a = self.ctx.gym.set_dof_state_tensor_indexed(self.ctx.sim, self.dyn.dof_state_desc, indices, len(env_idx))
        b = self.ctx.gym.set_actor_root_state_tensor_indexed(self.ctx.sim, self.dyn.root_state_desc, indices, len(env_idx))

        self.obs.history.reset(env_idx = env_idx)

        self.dyn.update()
        return self.obs.get(env_idx)[0]

    def update_contact_states(self):
        contact_state = self.dyn.get_feet_collsions()

        self.takeoff_time[contact_state] += self.ctx.dt # (new_time - self.step_timer)
        self.takeoff_time[~contact_state] = 0

        self.touchdown_time[contact_state] = 0
        self.touchdown_time[~contact_state] += self.ctx.dt # (new_time - self.step_timer)

    def compute_rewards(self, obs, target_dof_pos):
        mag = lambda mat, dim = -1: torch.sum(mat * mat, dim = dim)

        rot = obs[:, :4]
        ang_vel = obs[:, 4:7]
        dof_pos = obs[:, 7:19]
        dof_vel = obs[:, 19:31]
        t_dof_pos_t1 = obs[:, 31:43]
        t_dof_pos_t2 = obs[:, 43:55]
        hist_dof_pos = obs[:, 55:91].view(self.ctx.num_envs, 3, 12)
        hist_dof_vel = obs[:, 91:127].view(self.ctx.num_envs, 3, 12)
        rel_feet_pos = obs[:, 127:139].view(self.ctx.num_envs, 3, 4)
        vel_cmd = obs[:, 139:143]

        cur_feet_pos = self.dyn.get_feet_position() # (num_envs, 3, 4)
        cur_feet_vel = self.dyn.get_feet_velocity() # (num_envs, 3, 4)
        #contact_prob = self.contact_state.type(torch.FloatTensor) # (num_envs, 4)
        cur_lin_vel = self.dyn.get_linear_velocity()
        cur_ang_vel = self.dyn.get_angular_velocity()
        cur_joint_torque = self.dyn.get_joint_forces()

        r_v = K.k_v * torch.exp(-1 * mag(vel_cmd[:, :2] - cur_lin_vel[:, :2]))
        r_w = torch.zeros(self.ctx.num_envs).cuda() # K.k_w * torch.exp(-1.5 * mag(vel_cmd[:, 2:] - cur_ang_vel[:, 2:]))

        maxes = torch.where(self.takeoff_time > self.touchdown_time, self.takeoff_time, self.touchdown_time)
        r_air = K.k_a * torch.where(maxes > 0.25, 0., maxes.type(torch.DoubleTensor))
        r_air = torch.sum(r_air, dim = -1).cuda()


        norm_feet_vel = mag(cur_feet_vel[:, :, :2])
        #r_slip = K.k_slip * norm_feet_vel
        #r_slip[~self.contact_state] = 0
        r_slip = torch.zeros(self.ctx.num_envs).cuda() # torch.sum(r_slip, dim = -1)

        delta_cl = mag(cur_feet_pos[:, :, 2:] - K.desired_feet_height)
        feet_vel_quart = norm_feet_vel**0.25
        r_cl = K.k_cl * delta_cl * feet_vel_quart
        r_cl = torch.sum(r_cl, dim = -1)

        angle = tgm.quaternion_to_angle_axis(rot)
        rot_error = np.pi - angle[:, 2]
        r_ori = K.k_ori * torch.abs(rot_error)
        r_t = torch.zeros(self.ctx.num_envs).cuda() # K.k_t * mag(cur_joint_torque)
        r_q = K.k_q * mag(dof_pos - self.ctx.default_dof_pos)
        r_qdot = K.k_qdot * mag(dof_vel)
        r_qddot = K.k_qddot * mag(dof_vel - hist_dof_vel[:, 0, :])
        r_s1 = K.k_s1 * mag(target_dof_pos - t_dof_pos_t1)
        r_s2 = K.k_s2 * mag(target_dof_pos - 2 * t_dof_pos_t1 + t_dof_pos_t2)

        r_base = torch.zeros(self.ctx.num_envs).cuda()  #K.k_base * (0.8 * cur_lin_vel[:, 2] + 0.2 * torch.abs(cur_ang_vel[:, 0]) + 0.2 * cur_ang_vel[:, 1])

        pos_reward = r_v + r_w + r_air
        neg_reward = r_cl + r_ori #+ r_slip + r_t + r_qddot + r_s1 + r_s2 + r_base # r_q + r_qdot +

        total_reward = pos_reward * torch.exp(0.2 * neg_reward)

        terms = torch.mean(torch.stack([total_reward, pos_reward, neg_reward, r_v, r_w, r_air, r_slip, r_cl, r_t, r_q, r_qdot, r_qddot, r_s1, r_s2, r_base, r_ori]), dim = 1)
        names = ['total', 'pos', 'neg', 'r_v', 'r_w', 'r_air', 'r_slip', 'r_cl', 'r_t', 'r_q', 'r_qdot', 'r_qddot', 'r_s1', 'r_s2', 'r_base', 'r_ori']

        return total_reward, terms, names

     

    def step_simulation(self, target_dof_pos):
        self.dyn.update()
        self.update_contact_states()

        # do stuff
        observations, loss = self.obs.get()
        rewards, terms, names = self.compute_rewards(observations, target_dof_pos)

        self.obs.history.step(target_dof_pos, self.dyn.get_dof_position(), self.dyn.get_dof_velocity())
        self.ep_lens += 1

        dones = self.dyn.get_body_collisions()
        ep_dones = self.ep_lens > self.ctx.max_ep_steps

        reset_idx = []
        for term_idx in torch.where(ep_dones)[0]:
            reset_idx.append(term_idx.item())    

        for term_idx in torch.where(dones)[0]:
            rewards[term_idx] = -10
            reset_idx.append(term_idx.item())

        if len(reset_idx) > 0:
            observations[reset_idx, :] = self.reset(reset_idx)

        infos = [{} for _ in range(self.ctx.num_envs)]
        infos[0]['terms'] = terms
        infos[0]['names'] = names
        infos[0]['loss'] = loss.item()
        return observations, rewards, torch.logical_or(dones, ep_dones.cuda()), infos