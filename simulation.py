from isaacgym import gymapi, gymutil, gymtorch
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import time
import collections

from isaacgym.torch_utils import *

# TODO: make setters not modify original views?
# TODO: support for batched updates to avoid calling set_actor_XXX multiple times
# TODO: better support for getting rigid bodies, currently hardcoding support for feet
# TODO: cache feet positions and update every step
class RobotDynamics:
    def __init__(self, gym, sim):
        self.gym = gym
        self.sim = sim
        self.num_envs = gym.get_env_count(sim)

        self._root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        self._dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
        self._net_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        self._joint_forces = gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim))

        root_state_view = self._root_state.view(-1, 13)
        self.root_position = root_state_view[:, :3]    
        self.root_rotation =  root_state_view[:, 3:7] 
        self.root_lin_vel = root_state_view[:, 7:10]
        self.root_ang_vel = root_state_view[:, 10:]
        
        dof_state_view = self._dof_state.view(self.num_envs, -1, 2)
        self.dof_pos = dof_state_view[:, :, 0]
        self.dof_vel = dof_state_view[:, :, 1]

        self.net_forces = self._net_forces.view(self.num_envs, -1, 3)
        self.joint_forces = self._joint_forces.view(self.num_envs, -1)


    def step(self, step_physics = True):
        if step_physics:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)        

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    def get_position(self):
        return self.root_position

    def get_joint_forces(self):
        return self.joint_forces

    def set_position(self, pos, env_idx = None):
        if env_idx is None:
            self.root_position[:] = pos
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
        else:
            env = self.gym.get_env(self.sim, env_idx)
            self.step(False)
            actor_idx = gymtorch.unwrap_tensor(torch.IntTensor([env_idx]))
            self.root_position[env_idx, :] = pos
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
            #self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_state), actor_idx, 1)

    def get_rotation(self):
        return R.from_quat(self.root_rotation)

    def set_rotation(self, rot, env_idx = None):
        if env_idx is None:
            self.root_rotation[:] = torch.from_numpy(rot.as_quat())
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
        else:
            env = self.gym.get_env(self.sim, env_idx)
            self.step(False)
            actor_idx = gymtorch.unwrap_tensor(torch.IntTensor([env_idx]))
            self.root_rotation[env_idx, :] = torch.from_numpy(rot.as_quat())
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
            #self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_state), actor_idx, 1)

    def get_linear_velocity(self):
        return quat_rotate_inverse(self.root_rotation, self.root_lin_vel)

    def set_linear_velocity(self, lin_vel, env_idx = None):
        if env_idx is None:
            self.root_lin_vel[:] = lin_vel
        else:
            self.root_lin_vel[env_idx, :] = lin_vel

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))


    def get_angular_velocity(self):
        return quat_rotate_inverse(self.root_rotation, self.root_ang_vel)

    def set_angular_velocity(self, ang_vel, env_idx = None):
        if env_idx is None:
            self.root_ang_vel[:] = ang_vel
        else:
            self.root_ang_vel[env_idx, :] = ang_vel

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

    def get_dof_position(self, joint_idx = None):
        if joint_idx is None:
            return self.dof_pos
        else:
            return self.dof_pos[:, joint_idx]

    def set_dof_position(self, dof_pos, env_idx = None):
        if env_idx is None:
            self.dof_pos[:] = dof_pos
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        else:
            env = self.gym.get_env(self.sim, env_idx)
            self.step(False)
            actor_idx = gymtorch.unwrap_tensor(torch.IntTensor([env_idx]))
            self.dof_pos[env_idx, :] = dof_pos
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
            #self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), actor_idx, 1)

    def get_dof_velocity(self, joint_idx = None):
        if joint_idx is None:
            return self.dof_vel
        else:
            return self.dof_vel[:, joint_idx]

    def set_dof_velocity(self, dof_vel, env_idx = None):
        if env_idx is None:
            self.dof_vel[:] = dof_vel
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        else:
            env = self.gym.get_env(self.sim, env_idx)
            self.step(False)
            actor_idx = gymtorch.unwrap_tensor(torch.IntTensor([env_idx]))
            self.dof_vel[env_idx, :] = dof_vel
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
            #self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), actor_idx, 1)


    def get_force_tensor(self, body_idx = None):
        if body_idx is None:
            return self.net_forces
        else:
            return self.net_forces[:, body_idx]

    # TODO: Make this more efficient
    def get_feet_position(self, relative = False):
        feet_pos = torch.zeros((self.num_envs, 3, 4))
        feet_idx = [5, 9, 13, 17]

        body_states = self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_POS)
        num_rb = len(body_states) // self.num_envs

        # TODO: find a much better way to do this
        for i in range(self.num_envs):
            base_pos = self.root_position[i] * relative
            for arr_pos, _feet_idx in enumerate(feet_idx):
                arr = body_states[i * num_rb + _feet_idx]['pose']['p']
                arr = np.array([a for a in arr]) # why is this such a weird format :(
                feet_pos[i, :, arr_pos] = torch.from_numpy(arr) - base_pos

        return feet_pos

    def get_feet_velocity(self):
        feet_pos = torch.zeros((self.num_envs, 3, 4))
        feet_idx = [5, 9, 13, 17]

        body_states = self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_VEL)
        num_rb = len(body_states) // self.num_envs

        # TODO: find a much better way to do this
        for i in range(self.num_envs):
            for arr_pos, _feet_idx in enumerate(feet_idx):
                arr = body_states[i * num_rb + _feet_idx]['vel']['linear']
                arr = np.array([a for a in arr]) # why is this such a weird format :(
                feet_pos[i, :, arr_pos] = quat_rotate_inverse(self.root_rotation[i:i+1], torch.from_numpy(arr).unsqueeze(0))[0]

        return feet_pos

    def get_collisions(self, idx_to_check, collision_thresh):
        contact_forces = torch.norm(self.net_forces[:, idx_to_check, :], dim = 2) # (num_envs, len(idx_to_check))
        collision_idx = contact_forces > collision_thresh # (num_envs, len(idx_to_check))
        is_collision = torch.any(collision_idx, dim = 1) # (num_envs)
        return is_collision, collision_idx

    def get_feet_collsions(self, collision_thresh = 1.0):
        return self.get_collisions([5, 9, 13, 17], collision_thresh)

    def get_body_collisions(self, collision_thresh = 1.0):
        body_contacts = [0, 1, 2, 3, 6, 7, 10, 11, 14, 15]

        return self.get_collisions(body_contacts, collision_thresh)
    
class SimHelper:
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.sim = self.create_sim()

    def get_gym(self):
        return self.gym, self.sim

    def create_sim(self):
        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 8
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # set Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5

        # create sim with these parameters
        return self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def create_planes(self, env_shape):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

        lower = gymapi.Vec3()
        upper = gymapi.Vec3(env_shape[0], env_shape[1], env_shape[2])

        return lower, upper

    def create_envs(self, asset_path, asset_name, num_envs, env_shape, props = {}):
        asset_handle = self.gym.load_asset(self.sim, asset_path, asset_name)
        lower, upper = self.create_planes(env_shape)

        envs, actors = [], []
        envs_per_row = int(np.sqrt(num_envs))
        for idx in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, envs_per_row)

            identity_quat = R.identity().as_quat()
            pos = gymapi.Transform(p = gymapi.Vec3(0, 0, 0.38), r = gymapi.Quat(*identity_quat))
            actor = self.gym.create_actor(env, asset_handle, pos, f"{asset_name}_{idx}", idx)
            self.gym.enable_actor_dof_force_sensors(env, actor)

            # TODO: figure out how to do this with asset_props parameter, its a numpy array instead of a dict?
            # asset_props = {**asset_props, **props}
            asset_props = self.gym.get_asset_dof_properties(asset_handle)
            asset_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            asset_props["stiffness"].fill(17)
            asset_props["damping"].fill(0.5)
            self.gym.set_actor_dof_properties(env, actor, asset_props)

            envs.append(env)
            actors.append(actor)

        return asset_handle, envs, actors

class HistoryBuffer:
    def __init__(self, num_envs, default_dof_pos, default_dof_vel):
        self.num_envs = num_envs
        self.timers = torch.zeros(num_envs) # time.perf_counter()
        self.prev_timers = self.timers.clone()

        self.latest = [collections.deque(maxlen = 2) for _ in range(num_envs)]
        self.pos_history = [collections.deque(maxlen = 3) for _ in range(num_envs)]
        self.vel_history = [collections.deque(maxlen = 3) for _ in range(num_envs)]

        self.default_dof_pos = default_dof_pos
        self.default_dof_vel = default_dof_vel

        for i in range(3):
            for j in range(self.num_envs):
                self.latest[j].append(default_dof_pos)
                self.pos_history[j].append(default_dof_pos)
                self.vel_history[j].append(default_dof_vel)

    def step(self, des_dof_pos, dof_pos, dof_vel):
        self.timers += 1

        for i, (des_dp, dp, dv) in enumerate(zip(des_dof_pos, dof_pos, dof_vel)):
            self.latest[i].append(des_dp)

            cur_counter = time.perf_counter()
            if self.timers[i] - self.prev_timers[i] > 12:
                self.pos_history[i].append(dp)
                self.vel_history[i].append(dv)
                self.prev_timers[i] = self.timers[i]

    def get_dof_pos(self, env_idx = None):
        if env_idx is None:
            out = torch.zeros(2, self.num_envs, 12)
            for i in range(self.num_envs):
                for j in range(2):
                    out[j, i, :] = self.latest[i][j][:]
            return out
            #return torch.Tensor(np.array(self.latest)).view(2, self.num_envs, 12)
        else:
            return torch.Tensor(np.array(self.latest[env_idx])).view(2, 12)

    def get_dof_pos_history(self, env_idx = None):
        if env_idx is None:
            out = torch.zeros(3, self.num_envs, 12)
            for i in range(self.num_envs):
                for j in range(3):
                    out[j, i, :] = self.pos_history[i][j][:]
            return out
            #return torch.Tensor(np.array(self.pos_history)).view(3, self.num_envs, 12)
        else:
            return torch.Tensor(np.array(self.pos_history[env_idx])).view(3, 12)

    def get_dof_vel_history(self, env_idx = None):
        if env_idx is None:
            out = torch.zeros(3, self.num_envs, 12)
            for i in range(self.num_envs):
                for j in range(3):
                    out[j, i, :] = self.vel_history[i][j][:]
            return out
            #return torch.Tensor(np.array(self.vel_history)).view(3, self.num_envs, 12)
        else:
            return torch.Tensor(np.array(self.vel_history[env_idx])).view(3, 12)

    def reset(self, env_idx = None):
        if env_idx is None:
            for i in range(self.num_envs):
                for j in range(3):
                    self.latest[i].append(self.default_dof_pos)
                    self.pos_history[i].append(self.default_dof_pos)
                    self.vel_history[i].append(self.default_dof_vel)
                    self.timers[i] = 0
        else:
            self.timers[env_idx] = 0
            for i in range(3):
                self.latest[env_idx].append(self.default_dof_pos)
                self.pos_history[env_idx].append(self.default_dof_pos)
                self.vel_history[env_idx].append(self.default_dof_vel)
                    