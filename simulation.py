from isaacgym import gymapi, gymutil, gymtorch
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import time
import collections

from isaacgym.torch_utils import *

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
        sim_params.use_gpu_pipeline = True

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 8
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0


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
            pos = gymapi.Transform(p = gymapi.Vec3(0, 0, 0.28), r = gymapi.Quat(*identity_quat))
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