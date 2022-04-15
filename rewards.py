
class SEReward:
	def __init__(self, ctx, dyn, obs):
		self.ctx = ctx
		self.dyn = dyn
		self.obs = obs

		self.mag = lambda mat, dim = -1: torch.sum(mat * mat, dim = dim)

	def reward_linvel(self, scale = 3):
		vel_cmd = self.obs[:, 139:143]
		cur_lin_vel = self.dyn.get_linear_velocity()

		return scale * torch.exp(-1 * self.mag(vel_cmd[:, :2] - cur_lin_vel[:, :2]))

	def reward_angvel(self, scale = 3):
		vel_cmd = self.obs[:, 139:143]
		cur_ang_vel = self.dyn.get_angular_velocity()

		return torch.zeros(self.ctx.num_envs) # scale * torch.exp(-1.5 * self.mag(vel_cmd[:, 2:] - cur_ang_vel[:, 2:]))

	def reward_airtime(self, scale = 0.3):
		maxes = torch.where(self.takeoff_time > self.touchdown_time, self.takeoff_time, self.touchdown_time)
        r_air = K.k_a * torch.where(maxes > 0.25, 0., maxes.type(torch.DoubleTensor))
        r_air = torch.sum(r_air, dim = -1)

	def reward_slip(self, scale = -0.08):
		norm_feet_vel = mag(cur_feet_vel[:, :2, :].transpose(1, 2))
        r_slip = K.k_slip * norm_feet_vel
        r_slip[~self.contact_state] = 0
        r_slip = torch.sum(r_slip, dim = -1)

	def reward_clearance(self, scale = -15):
		delta_cl = mag(cur_feet_pos[:, 2:, :] - K.desired_feet_height, dim = 1)
        feet_vel_quart = norm_feet_vel**0.25
        r_cl = K.k_cl * delta_cl * feet_vel_quart
        r_cl = torch.sum(r_cl, dim = -1)

	def reward_orientation(self, scale = -3.0):
        rot_error = np.pi - torch.Tensor(rot.as_euler('xyz', degrees = False)[:, 2])
        r_ori = K.k_ori * torch.abs(rot_error)

	def reward_joint_torque(self, scale = -6e-4):
		r_t = torch.zeros(self.num_envs) # K.k_t * mag(cur_joint_torque)

	def reward_joint_pos(self, scale = -0.75):
		r_q = K.k_q * mag(dof_pos - self.default_dof_pos)

	def reward_joint_speed(self, scale = -6e-4):
		 r_qdot = K.k_qdot * mag(dof_vel)

	def reward_joint_acc(self, scale = -0.02):
		r_qddot = K.k_qddot * mag(dof_vel - hist_dof_vel[:, 0, :])

	def reward_smooth_1(self, scale = -2.5):
		r_s1 = K.k_s1 * mag(target_dof_pos - t_dof_pos_t1)

	def reward_smooth_2(self, scale = -1.2):
		r_s2 = K.k_s2 * mag(target_dof_pos - 2 * t_dof_pos_t1 + t_dof_pos_t2)

	def reward_base_motion(self, scale = -1.5):
		r_base = torch.zeros(self.num_envs)  #K.k_base * (0.8 * cur_lin_vel[:, 2] + 0.2 * torch.abs(cur_ang_vel[:, 0]) + 0.2 * cur_ang_vel[:, 1])

	def get(self, scale = ):
		pos_reward = r_v + r_w + r_air
        neg_reward = r_cl + r_ori + #r_slip + r_t + r_qddot + r_s1 + r_s2 + r_base # r_q + r_qdot

        total_reward = pos_reward * torch.exp(0.2 * neg_reward)

        terms = torch.mean(torch.stack([total_reward, pos_reward, neg_reward, r_v, r_w, r_air, r_slip, r_cl, r_t, r_q, r_qdot, r_qddot, r_s1, r_s2, r_base, r_ori]), dim = 1)
        names = ['total', 'pos', 'neg', 'r_v', 'r_w', 'r_air', 'r_slip', 'r_cl', 'r_t', 'r_q', 'r_qdot', 'r_qddot', 'r_s1', 'r_s2', 'r_base', 'r_ori']

        return total_reward, terms, names


	def compute_rewards(self, obs, target_dof_pos):
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