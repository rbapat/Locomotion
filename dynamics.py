from isaacgym import gymapi, gymutil, gymtorch
from stable_baselines3.common.vec_env import VecEnv
from scipy.spatial.transform import Rotation as R
import torchgeometry as tgm
import numpy as np
import torch
import torch.nn as nn
import time
import gym

class AgentDynamics:
    def __init__(self, ctx):
        self.ctx = ctx
        self.gym, self.sim = ctx.gym, ctx.sim

        self.root_state_desc = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(self.root_state_desc)

        self.dof_state_desc = self.gym.acquire_dof_state_tensor(self.sim)
        self._dof_state = gymtorch.wrap_tensor(self.dof_state_desc)

        self._rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self._net_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self._joint_forces = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim))

        root_state_view = self._root_state.view(self.ctx.num_envs, 13)
        self.root_position = root_state_view[:, :3]    
        self.root_rotation =  root_state_view[:, 3:7] 
        self.root_lin_vel = root_state_view[:, 7:10]
        self.root_ang_vel = root_state_view[:, 10:]

        self.feet_idx = [5, 9, 13, 17]
        self.body_idx = [0, 1, 2, 3, 6, 7, 10, 11, 14, 15]


        dof_state_view = self._dof_state.view(self.ctx.num_envs, self.ctx.num_dof, 2)
        self.dof_pos = dof_state_view[:, :, 0]
        self.dof_vel = dof_state_view[:, :, 1]

        rb_view = self._rb_states.view(self.ctx.num_envs, self.ctx.num_rb, 13)
        self.rb_pos = rb_view[:, :, :3]
        self.rb_rot = rb_view[:, :, 3:7]
        self.rb_lin_vel = rb_view[:, :, 7:10]
        self.rb_ang_vel = rb_view[:, :, 10:]

        self.net_forces = self._net_forces.view(self.ctx.num_envs, self.ctx.num_rb, 3)
        self.joint_forces = self._joint_forces.view(self.ctx.num_envs, self.ctx.num_dof)

    def update(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def get_joint_forces(self):
        return self.joint_forces

    def get_position(self):
        return self.root_position

    def get_rotation(self):
        return self.root_rotation

    def get_linear_velocity(self):
        return self.root_lin_vel

    def get_angular_velocity(self):
        return self.root_ang_vel

    def get_dof_position(self):
        return self.dof_pos

    def get_dof_velocity(self):
        return self.dof_vel

    # TODO: Make this more efficient
    def get_feet_position(self, relative = False):
        feet_pos = self.rb_pos[:, self.feet_idx, :] # (num_envs, 4, 3)
        if relative:
            for i in range(3):
                feet_pos[:, i, :] -= self.root_position
        
        return feet_pos

    def get_feet_velocity(self):
        feet_pos = self.rb_lin_vel[:, self.feet_idx, :] # (num_envs, 4, 3)
        
        return feet_pos
        '''
        feet_pos = torch.zeros((self.ctx.num_envs, 3, 4))

        body_states = self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_POS)
        num_rb = len(body_states) // self.ctx.num_envs

        # TODO: find a much better way to do this
        for i in range(self.ctx.num_envs):
            for arr_pos, _feet_idx in enumerate(self.feet_idx):
                arr = body_states[i * num_rb + _feet_idx]['pose']['p']
                arr = np.array([a for a in arr]) # why is this such a weird format :(
                dwad = torch.from_numpy(arr).float().cuda()
                feet_pos[i, :, arr_pos] = dwad

                if relative:
                    feet_pos[i, :, arr_pos] -= self.root_position[i]

        return feet_pos
        '''

    def get_collisions(self, idx_to_check, collision_thresh):
        contact_forces = torch.norm(self.net_forces[:, idx_to_check, :], dim = 2) # (num_envs, len(idx_to_check))
        collision_idx = contact_forces > collision_thresh # (num_envs, len(idx_to_check))
        is_collision = torch.any(collision_idx, dim = 1) # (num_envs)
        return is_collision, collision_idx

    def get_feet_collsions(self, collision_thresh = 1.0):
        return self.get_collisions(self.feet_idx, collision_thresh)[1]

    def get_body_collisions(self, collision_thresh = 1.0):
        return self.get_collisions(self.body_idx, collision_thresh)[0]