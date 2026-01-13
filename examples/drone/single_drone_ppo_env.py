"""
单无人机避障路径规划环境 - PPO版本 + CNN感知
观测包含两部分：
1. 自身状态：全局位置、速度、到目标距离、目标方位角
2. 障碍物感知：3x3x3立方体网格（局部感知）
"""
import torch
import math
import copy
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class SingleDronePPOEnv:
    """单无人机PPO环境 - CNN+MLP混合架构"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        # ==================== 基础配置 ====================
        self.num_envs = num_envs
        self.rendered_env_num = min(5, self.num_envs)

        # 观测空间
        self.num_state_obs = obs_cfg["num_state_obs"]  # 自身状态维度
        self.grid_size = obs_cfg.get("grid_size", 3)  # 障碍物网格大小
        self.grid_resolution = env_cfg.get("grid_resolution", 1.0)  # 每个网格单元的大小(米)

        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # 4个电机
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # ==================== 创建仿真场景 ====================
        if show_viewer:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                viewer_options=gs.options.ViewerOptions(
                    max_FPS=env_cfg["max_visualize_FPS"],
                    camera_pos=(5.0, 0.0, 5.0),
                    camera_lookat=(0.0, 0.0, 1.0),
                    camera_fov=50,
                ),
                vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                ),
                show_viewer=True,
            )
        else:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                ),
                show_viewer=False,
            )

        self.scene.add_entity(gs.morphs.Plane())

        # ==================== 添加障碍物 ====================
        self.obstacles = []
        obstacle_positions = env_cfg.get("obstacle_positions", [])
        obstacle_radius = env_cfg.get("obstacle_radius", 0.12)
        obstacle_height = env_cfg.get("obstacle_height", 2.5)

        for pos in obstacle_positions:
            if show_viewer:
                obstacle = self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        pos=pos,
                        radius=obstacle_radius,
                        height=obstacle_height,
                        fixed=True,
                        collision=True,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.3, 0.8)),
                    ),
                )
            else:
                obstacle = None
            self.obstacles.append({
                "entity": obstacle,
                "pos": torch.tensor(pos, device=gs.device),
                "radius": obstacle_radius,
                "height": obstacle_height
            })

        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.4)
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.18)

        # ==================== 添加单架无人机 ====================
        self.drone_init_position = env_cfg.get("drone_init_position", [0.0, -2.5, 0.8])
        self.drone_goal_position = env_cfg.get("drone_goal_position", [0.0, 2.5, 0.8])

        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # 目标点可视化
        if env_cfg.get("visualize_target", False):
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj", scale=0.08, pos=self.drone_goal_position,
                    fixed=True, collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.2, 0.2)),
                ),
            )

        # ==================== 添加录制相机 ====================
        if env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(5.0, 0.0, 5.0),
                lookat=(0.0, 0.0, 1.0),
                fov=50,
                GUI=False,
            )
        else:
            self.cam = None

        self.scene.build(n_envs=num_envs)

        # ==================== 初始化奖励函数 ====================
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ==================== 初始化状态缓冲区 ====================
        # 观测维度：state(19) + flattened_grid(27) = 46
        self.num_obs = self.num_state_obs + (self.grid_size ** 3)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.command = torch.tensor(self.drone_goal_position, device=gs.device, dtype=gs.tc_float).unsqueeze(0).expand(self.num_envs, -1)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # 无人机状态
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        # 目标相关
        self.rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.dist_to_target = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.target_yaw = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _compute_obstacle_grid(self):
        """
        计算3x3x3障碍物感知网格
        网格中心是无人机自身，每个格子表示该方位是否有障碍物
        1 = 有障碍物, 0 = 无障碍物
        """
        grid = torch.zeros((self.num_envs, self.grid_size, self.grid_size, self.grid_size),
                          device=gs.device, dtype=gs.tc_float)

        if len(self.obstacles) == 0:
            return grid

        # 网格偏移（相对于无人机）
        half_size = self.grid_size // 2

        for env_idx in range(self.num_envs):
            drone_pos = self.base_pos[env_idx]

            for obs in self.obstacles:
                obs_pos = obs["pos"]
                obs_radius = obs["radius"]

                # 计算相对位置
                rel_pos = obs_pos - drone_pos

                # 转换为网格坐标
                grid_x = int((rel_pos[0] / self.grid_resolution) + half_size)
                grid_y = int((rel_pos[1] / self.grid_resolution) + half_size)
                grid_z = int((rel_pos[2] / self.grid_resolution) + half_size)

                # 检查是否在网格范围内
                if (0 <= grid_x < self.grid_size and
                    0 <= grid_y < self.grid_size and
                    0 <= grid_z < self.grid_size):

                    # 如果障碍物足够近，标记该网格
                    dist = torch.norm(rel_pos)
                    if dist < (obs_radius + self.grid_resolution):
                        grid[env_idx, grid_x, grid_y, grid_z] = 1.0

        return grid

    def _compute_target_yaw(self):
        """
        计算目标方位角：从无人机当前朝向到目标方向的角度
        返回值范围：[-π, π]
        """
        # 获取无人机当前朝向（yaw角）
        current_yaw = self.base_euler[:, 2] * math.pi / 180.0  # 转换为弧度

        # 计算目标方向
        target_direction = self.rel_pos[:, :2]  # 只考虑xy平面
        target_yaw = torch.atan2(target_direction[:, 1], target_direction[:, 0])

        # 计算相对角度
        relative_yaw = target_yaw - current_yaw

        # 归一化到[-π, π]
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))

        return relative_yaw

    def step(self, actions):
        """执行一步仿真"""
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # 设置动作
        self.drone.set_propellels_rpm((1 + self.actions * 0.8) * 14468.429183500699)

        self.scene.step()
        self.episode_length_buf += 1

        # 更新无人机状态
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()

        self.base_euler[:] = quat_to_xyz(
            transform_quat_by_quat(
                self.inv_base_init_quat.unsqueeze(0).expand(self.num_envs, -1),
                self.base_quat,
            ),
            rpy=True, degrees=True,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        self.last_rel_pos[:] = self.rel_pos[:]
        self.rel_pos = self.command - self.base_pos
        self.dist_to_target = torch.norm(self.rel_pos, dim=1)
        self.target_yaw = self._compute_target_yaw()

        # 计算最近障碍物距离
        self.min_obstacle_dist = self._get_min_obstacle_distance()

        # ==================== 终止条件 ====================
        crash = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            | (self.min_obstacle_dist < self.obstacle_collision_distance)
        )

        success = self.dist_to_target < self.env_cfg["at_target_threshold"]

        self.crash_condition = crash
        self.success_condition = success
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | crash | success

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ==================== 构建观测 ====================
        # 自身状态: [全局位置(3), 速度(3), 到目标距离(1), 目标方位角(1), 姿态四元数(4), 角速度(3), 上一动作(4)]
        state_obs = torch.cat([
            self.base_pos * self.obs_scales["pos"],  # 3
            torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),  # 3
            (self.dist_to_target * self.obs_scales["dist"]).unsqueeze(-1),  # 1
            (self.target_yaw * self.obs_scales["yaw"]).unsqueeze(-1),  # 1
            self.base_quat,  # 4
            torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),  # 3
            self.last_actions,  # 4
        ], dim=-1)  # 总共19维

        # 障碍物网格：3x3x3 -> 展平为27维
        obstacle_grid = self._compute_obstacle_grid()
        obstacle_grid_flat = obstacle_grid.reshape(self.num_envs, -1)  # (num_envs, 27)

        # 拼接观测：19 + 27 = 46维
        self.obs_buf = torch.cat([state_obs, obstacle_grid_flat], dim=-1)

        self.last_actions[:] = self.actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _get_min_obstacle_distance(self):
        """计算与最近障碍物的距离"""
        min_dist = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float) * 100.0
        if len(self.obstacles) == 0:
            return min_dist

        drone_pos_xy = self.base_pos[:, :2]
        for obs in self.obstacles:
            obs_pos_xy = obs["pos"][:2].unsqueeze(0)
            dist = torch.norm(drone_pos_xy - obs_pos_xy, dim=1) - obs["radius"]
            min_dist = torch.minimum(min_dist, dist)
        return min_dist

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        """重置指定环境"""
        if len(envs_idx) == 0:
            return

        init_pos = torch.tensor(self.drone_init_position, device=gs.device)
        self.base_pos[envs_idx] = init_pos
        self.last_base_pos[envs_idx] = init_pos
        self.base_quat[envs_idx] = self.base_init_quat

        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ==================== 奖励函数 ====================
    def _reward_target(self):
        """目标奖励"""
        last_dist = torch.norm(self.last_rel_pos, dim=1)
        dist_reduction = last_dist - self.dist_to_target

        target_rew = dist_reduction * 10.0
        target_rew -= self.dist_to_target * 0.1
        target_rew += torch.where(self.dist_to_target < 2.0, torch.ones_like(self.dist_to_target) * 2.0, torch.zeros_like(self.dist_to_target))
        target_rew += torch.where(self.dist_to_target < 1.0, torch.ones_like(self.dist_to_target) * 5.0, torch.zeros_like(self.dist_to_target))
        target_rew[self.success_condition] += 100.0

        return target_rew

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_obstacle(self):
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew

        danger_dist = self.obstacle_safe_distance * 0.6
        close_mask = self.min_obstacle_dist < danger_dist
        obstacle_rew[close_mask] = -(danger_dist - self.min_obstacle_dist[close_mask]) / danger_dist
        return obstacle_rew

    def _reward_progress(self):
        progress_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # Y方向进度
        y_progress = self.base_pos[:, 1] - self.last_base_pos[:, 1]
        progress_rew += y_progress * 20.0

        # 高度保持
        height = self.base_pos[:, 2]
        height_good = (height > 0.4) & (height < 1.5)
        progress_rew += torch.where(height_good, torch.ones_like(height) * 1.0, -torch.ones_like(height) * 0.5)

        # 姿态稳定
        roll = torch.abs(self.base_euler[:, 0])
        pitch = torch.abs(self.base_euler[:, 1])
        stable = (roll < 30) & (pitch < 30)
        progress_rew += torch.where(stable, torch.ones_like(roll) * 0.5, -torch.ones_like(roll) * 1.0)

        return progress_rew

    def _reward_alive(self):
        alive_rew = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        alive_rew[self.crash_condition] = 0
        return alive_rew
