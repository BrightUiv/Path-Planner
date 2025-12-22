"""
多无人机避障路径规划环境 - MAPPO版本
MAPPO采用集中式训练分布式执行(CTDE)架构：
- 观测：每架无人机独立观测（局部信息）
- 动作：每架无人机独立决策
- Critic：使用全局状态信息进行价值估计
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
    """
    生成指定范围内的随机浮点数张量
    
    参数:
        lower: 随机数下界
        upper: 随机数上界
        shape: 输出张量形状
        device: 计算设备
    返回:
        [lower, upper) 范围内的随机张量
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class MultiDroneMAPPOEnv:
    """多无人机MAPPO环境 - 集中式训练分布式执行"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        """
        初始化多无人机MAPPO环境
        
        参数:
            num_envs: 并行环境数量
            env_cfg: 环境配置（无人机数量、终止条件等）
            obs_cfg: 观测配置（观测维度、缩放系数等）
            reward_cfg: 奖励配置（各奖励项权重）
            command_cfg: 命令配置（目标位置等）
            show_viewer: 是否显示可视化窗口
        """
        # ==================== 基础配置 ====================
        self.num_envs = num_envs
        self.num_drones = env_cfg.get("num_drones", 3)
        self.num_agents = self.num_drones  # MAPPO: 智能体数量
        self.rendered_env_num = min(5, self.num_envs)
        
        # MAPPO关键：每个智能体独立的观测和动作维度
        # 单架无人机局部观测：智能体ID(3) + 相对位置(3)+四元数(4)+线速度(3)+角速度(3)+动作(4) = 20维
        self.num_obs_per_agent = obs_cfg.get("num_obs_per_agent", 20)
        # 其他无人机相对位置信息：(num_drones-1) * 3 = 6维
        self.num_other_agents_obs = (self.num_drones - 1) * 3
        # 完整局部观测维度：20 + 6 = 26维
        self.num_obs = self.num_obs_per_agent + self.num_other_agents_obs
        # 全局状态维度（Critic使用）：所有无人机状态拼接
        self.num_state = self.num_obs * self.num_drones
        # 特权观测（本环境未使用）
        self.num_privileged_obs = None
        # 单架无人机动作维度：4个螺旋桨的转速控制
        self.num_actions = env_cfg["num_actions"]  # 4
        # 命令维度：目标位置的xyz坐标
        self.num_commands = command_cfg["num_commands"]
        # 计算设备（CPU/GPU）
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
        # 根据是否需要可视化来配置场景
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
            # 无头模式：完全禁用可视化，避免 EGL/OpenGL 错误
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,
                    enable_collision=True,
                    enable_joint_limit=True,
                ),
                show_viewer=False,
                show_FPS=False,
            )

        self.scene.add_entity(gs.morphs.Plane())

        # ==================== 添加障碍物 ====================
        self.obstacles = []
        obstacle_positions = env_cfg.get("obstacle_positions", [])
        obstacle_radius = env_cfg.get("obstacle_radius", 0.12)
        obstacle_height = env_cfg.get("obstacle_height", 2.5)

        for pos in obstacle_positions:
            # 无论是否可视化，都添加障碍物实体（用于碰撞检测）
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
                # 无头模式也添加障碍物，但不设置表面材质
                obstacle = self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        pos=pos,
                        radius=obstacle_radius,
                        height=obstacle_height,
                        fixed=True,
                        collision=True,
                    ),
                )
            self.obstacles.append({
                "entity": obstacle,
                "pos": torch.tensor(pos, device=gs.device),
                "radius": obstacle_radius
            })
        
        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.4)
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.18)
        self.drone_safe_distance = env_cfg.get("drone_safe_distance", 0.5)

        # ==================== 添加多架无人机 ====================
        self.drones = []
        self.drone_init_positions = env_cfg.get("drone_init_positions", [
            [-1.0, -2.5, 0.15], [0.0, -2.5, 0.15], [1.0, -2.5, 0.15],
        ])
        self.drone_goal_positions = env_cfg.get("drone_goal_positions", [
            [-1.0, 2.5, 0.15], [0.0, 2.5, 0.15], [1.0, 2.5, 0.15],
        ])
        
        drone_colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]  # RGB: 红、绿、蓝
        # 初始四元数 [w,x,y,z]=[1,0,0,0] 表示无旋转（单位四元数）
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)  # 逆四元数，用于坐标变换
        
        for i in range(self.num_drones):
            # 加载Crazyflie 2.x四旋翼无人机URDF模型
            drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
            self.drones.append(drone)

        # 目标点可视化（仅在可视化模式下）
        self.targets = []
        if show_viewer and env_cfg.get("visualize_target", False):
            for i, goal_pos in enumerate(self.drone_goal_positions):
                target = self.scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/sphere.obj", scale=0.08, pos=goal_pos,
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=drone_colors[i % len(drone_colors)]),
                    ),
                )
                self.targets.append(target)

        # ==================== 添加录制相机（仅在可视化模式下）====================
        self.cam = None
        if show_viewer and env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(5.0, 0.0, 5.0),
                lookat=(0.0, 0.0, 1.0),
                fov=50,
                GUI=False,
            )

        self.scene.build(n_envs=num_envs)

        # ==================== 初始化奖励函数 ====================
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)


        # ==================== 初始化状态缓冲区（MAPPO关键：多智能体结构）====================
        # 每个智能体的局部观测：(num_envs, num_agents, num_obs)
        self.obs_buf = torch.zeros((self.num_envs, self.num_agents, self.num_obs), device=gs.device, dtype=gs.tc_float)
        # 全局状态（Critic使用）：(num_envs, num_state)
        self.state_buf = torch.zeros((self.num_envs, self.num_state), device=gs.device, dtype=gs.tc_float)
        # 每个智能体的奖励：(num_envs, num_agents)
        self.rew_buf = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        # 环境重置标志
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        
        # 目标位置命令
        self.commands = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        # 每个智能体的动作：(num_envs, num_agents, num_actions)
        self.actions = torch.zeros((self.num_envs, self.num_agents, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # 每架无人机的状态
        self.base_pos = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, self.num_drones, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, self.num_drones, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        # 可用掩码（标记哪些智能体仍然活跃）
        self.available_actions_mask = torch.ones((self.num_envs, self.num_agents, self.num_actions), device=gs.device, dtype=torch.bool)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        """
        设置各无人机的目标位置
        
        参数:
            envs_idx: 需要重置目标的环境索引
        """
        for i, goal_pos in enumerate(self.drone_goal_positions):
            self.commands[envs_idx, i, 0] = goal_pos[0]
            self.commands[envs_idx, i, 1] = goal_pos[1]
            self.commands[envs_idx, i, 2] = goal_pos[2]


    def step(self, actions):
        """
        执行一步仿真（MAPPO版本）
        
        参数:
            actions: 所有智能体的动作张量，形状为 (num_envs, num_agents, num_actions)
        返回:
            obs_buf: 每个智能体的局部观测 (num_envs, num_agents, num_obs)
            state_buf: 全局状态 (num_envs, num_state)
            rew_buf: 每个智能体的奖励 (num_envs, num_agents)
            reset_buf: 重置标志缓冲区
            extras: 额外信息字典
        """
        # 裁剪动作到合法范围
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # 为每架无人机设置动作（将动作转换为螺旋桨转速）
        for i, drone in enumerate(self.drones):
            drone_actions = self.actions[:, i, :]  # (num_envs, num_actions)
            # 动作映射：[-1,1] -> [0.2, 1.8] * 悬停转速 = [2894, 26043] RPM
            # 14468.429... 是Crazyflie 2.x的悬停转速（hover RPM）
            drone.set_propellels_rpm((1 + drone_actions * 0.8) * 14468.429183500699)

        self.scene.step()
        self.episode_length_buf += 1

        # 更新每架无人机的状态（位置、姿态、速度等）
        self.last_base_pos[:] = self.base_pos[:]
        for i, drone in enumerate(self.drones):
            self.base_pos[:, i, :] = drone.get_pos()
            self.base_quat[:, i, :] = drone.get_quat()
            
            # 将四元数转换为欧拉角（roll, pitch, yaw）
            self.base_euler[:, i, :] = quat_to_xyz(
                transform_quat_by_quat(
                    self.inv_base_init_quat.unsqueeze(0).expand(self.num_envs, -1),
                    self.base_quat[:, i, :],
                ),
                rpy=True, degrees=True,
            )
            
            # 将世界坐标系速度转换为机体坐标系速度
            inv_quat_i = inv_quat(self.base_quat[:, i, :])
            self.base_lin_vel[:, i, :] = transform_by_quat(drone.get_vel(), inv_quat_i)
            self.base_ang_vel[:, i, :] = transform_by_quat(drone.get_ang(), inv_quat_i)

        # 更新相对位置（当前位置到目标的距离）
        self.last_rel_pos[:] = self.rel_pos[:]
        self.rel_pos = self.commands - self.base_pos

        # 计算与障碍物和其他无人机的最小距离
        self.min_obstacle_dist = self._get_min_obstacle_distance()
        self.min_drone_dist = self._get_min_drone_distance()

        # ==================== 终止条件检测 ====================
        crash_any = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        success_all = torch.ones((self.num_envs,), device=gs.device, dtype=torch.bool)
        # 每个智能体的坠毁状态
        self.agent_crash = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=torch.bool)
        # 每个智能体的成功状态
        self.agent_success = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=torch.bool)
        
        for i in range(self.num_drones):
            # 检测单架无人机是否坠毁（姿态过大、高度过低、碰撞障碍物）
            drone_crash = (
                (torch.abs(self.base_euler[:, i, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
                | (torch.abs(self.base_euler[:, i, 0]) > self.env_cfg["termination_if_roll_greater_than"])
                | (self.base_pos[:, i, 2] < self.env_cfg["termination_if_close_to_ground"])
                | (self.min_obstacle_dist[:, i] < self.obstacle_collision_distance)
            )
            self.agent_crash[:, i] = drone_crash
            crash_any = crash_any | drone_crash
            
            # 检测单架无人机是否到达目标
            drone_success = torch.norm(self.rel_pos[:, i, :], dim=1) < self.env_cfg["at_target_threshold"]
            self.agent_success[:, i] = drone_success
            success_all = success_all & drone_success

        # 检测无人机之间是否发生碰撞（任意两架距离小于阈值）
        drone_collision = self.min_drone_dist < self.env_cfg.get("drone_collision_distance", 0.3)
        crash_any = crash_any | drone_collision

        self.crash_condition = crash_any      # 任一无人机坠毁/碰撞
        self.success_condition = success_all  # 所有无人机都到达目标
        # 重置条件：超时 或 坠毁 或 全部成功
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | crash_any | success_all

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励（MAPPO：每个智能体独立奖励）====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]  # (num_envs, num_agents)
            self.rew_buf += rew
            # episode_sums 记录环境级别的平均奖励
            self.episode_sums[name] += rew.mean(dim=1)

        # ==================== 构建观测（MAPPO关键：局部观测 + 全局状态）====================
        self._compute_observations()
        
        self.last_actions[:] = self.actions[:]
        
        # extras中包含全局状态供Critic使用
        self.extras["observations"]["critic"] = self.state_buf
        self.extras["available_actions"] = self.available_actions_mask

        return self.obs_buf, self.state_buf, self.rew_buf, self.reset_buf, self.extras


    def _compute_observations(self):
        """
        计算每个智能体的局部观测和全局状态
        
        局部观测（每个智能体）：
            - 智能体ID (one-hot编码, 3维)
            - 到目标的相对位置 (3维)
            - 姿态四元数 (4维)
            - 线速度 (3维)
            - 角速度 (3维)
            - 上一步动作 (4维)
            - 其他无人机的相对位置 (6维)
            总计：26维
        
        全局状态（Critic使用）：
            - 所有智能体的局部观测拼接
            总计：26 * 3 = 78维
        """
        obs_list = []
        for i in range(self.num_agents):
            # 智能体ID (one-hot编码)：3维 - 让共享Actor能区分不同无人机
            agent_id = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
            agent_id[:, i] = 1.0
            
            # 基础局部观测：17维
            base_obs = torch.cat([
                agent_id,  # 3维 - 智能体身份标识
                torch.clip(self.rel_pos[:, i, :] * self.obs_scales["rel_pos"], -1, 1),  # 3
                self.base_quat[:, i, :],  # 4
                torch.clip(self.base_lin_vel[:, i, :] * self.obs_scales["lin_vel"], -1, 1),  # 3
                torch.clip(self.base_ang_vel[:, i, :] * self.obs_scales["ang_vel"], -1, 1),  # 3
                self.last_actions[:, i, :],  # 4
            ], dim=-1)
            
            # 其他无人机的相对位置：(num_agents-1) * 3 = 6维
            other_agents_rel_pos = []
            for j in range(self.num_agents):
                if i != j:
                    rel_to_other = (self.base_pos[:, j, :] - self.base_pos[:, i, :]) * self.obs_scales["rel_pos"]
                    other_agents_rel_pos.append(torch.clip(rel_to_other, -1, 1))
            
            # 拼接单个智能体的完整局部观测：20 + 6 = 26维
            agent_obs = torch.cat([base_obs] + other_agents_rel_pos, dim=-1)
            self.obs_buf[:, i, :] = agent_obs
            obs_list.append(agent_obs)
        
        # 全局状态：所有智能体观测拼接，26 * 3 = 78维
        self.state_buf = torch.cat(obs_list, dim=-1)

    def _get_min_obstacle_distance(self):
        """
        计算每架无人机与障碍物的最小距离（仅考虑XY平面，忽略高度）
        
        返回:
            min_dist: 形状为 (num_envs, num_drones) 的张量，表示到最近障碍物边缘的距离
        """
        min_dist = torch.ones((self.num_envs, self.num_drones), device=gs.device, dtype=gs.tc_float) * 100.0
        if len(self.obstacles) == 0:
            return min_dist
        for i in range(self.num_drones):
            drone_pos_xy = self.base_pos[:, i, :2]
            for obs in self.obstacles:
                obs_pos_xy = obs["pos"][:2].unsqueeze(0)
                # 计算XY平面距离并减去障碍物半径，得到到障碍物边缘的距离
                dist = torch.norm(drone_pos_xy - obs_pos_xy, dim=1) - obs["radius"]
                min_dist[:, i] = torch.minimum(min_dist[:, i], dist)
        return min_dist

    def _get_min_drone_distance(self):
        """
        计算无人机之间的最小距离（3D欧氏距离）
        
        返回:
            min_dist: 形状为 (num_envs,) 的张量，表示所有无人机对中的最小距离
        """
        min_dist = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float) * 100.0
        # 遍历所有无人机对 (i,j)，其中 i < j，避免重复计算
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = torch.norm(self.base_pos[:, i, :] - self.base_pos[:, j, :], dim=1)
                min_dist = torch.minimum(min_dist, dist)
        return min_dist

    def get_observations(self):
        """
        获取当前观测（MAPPO版本）
        
        返回:
            obs_buf: 每个智能体的局部观测 (num_envs, num_agents, num_obs)
            extras: 额外信息字典（包含全局状态）
        """
        self.extras["observations"]["critic"] = self.state_buf
        return self.obs_buf, self.extras

    def get_state(self):
        """
        获取全局状态（MAPPO Critic使用）
        
        返回:
            state_buf: 全局状态 (num_envs, num_state)
        """
        return self.state_buf

    def get_privileged_observations(self):
        """
        获取特权观测（本环境未使用）
        
        返回:
            None
        """
        return None


    def reset_idx(self, envs_idx):
        """
        重置指定环境
        
        参数:
            envs_idx: 需要重置的环境索引张量
        """
        if len(envs_idx) == 0:
            return

        # 重置每架无人机的位置和姿态
        for i, drone in enumerate(self.drones):
            init_pos = torch.tensor(self.drone_init_positions[i], device=gs.device)
            self.base_pos[envs_idx, i, :] = init_pos
            self.last_base_pos[envs_idx, i, :] = init_pos
            self.base_quat[envs_idx, i, :] = self.base_init_quat
            
            drone.set_pos(self.base_pos[envs_idx, i, :], zero_velocity=True, envs_idx=envs_idx)
            drone.set_quat(self.base_quat[envs_idx, i, :], zero_velocity=True, envs_idx=envs_idx)
            drone.zero_all_dofs_velocity(envs_idx)

        # 重置速度和动作缓冲区
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 重置可用动作掩码
        self.available_actions_mask[envs_idx] = True

        # 记录episode统计信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        """
        重置所有环境
        
        返回:
            obs_buf: 每个智能体的初始局部观测 (num_envs, num_agents, num_obs)
            state_buf: 初始全局状态 (num_envs, num_state)
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._compute_observations()
        return self.obs_buf, self.state_buf


    # ==================== 奖励函数（MAPPO：返回每个智能体的奖励）====================
    def _reward_target(self):
        """
        目标奖励：鼓励每个智能体接近各自的目标位置
        
        返回:
            target_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        target_rew = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        for i in range(self.num_agents):
            curr_dist = torch.norm(self.rel_pos[:, i, :], dim=1)
            last_dist = torch.norm(self.last_rel_pos[:, i, :], dim=1)
            
            # 距离缩减奖励：鼓励向目标移动（正向激励）
            dist_reduction = last_dist - curr_dist
            target_rew[:, i] += dist_reduction * 10.0
            # 距离惩罚：距离越远惩罚越大（持续压力）
            target_rew[:, i] -= curr_dist * 0.1
            # 接近奖励：进入2m范围给额外奖励
            target_rew[:, i] += torch.where(curr_dist < 2.0, torch.ones_like(curr_dist) * 2.0, torch.zeros_like(curr_dist))
            # 接近奖励：进入1m范围给更大奖励
            target_rew[:, i] += torch.where(curr_dist < 1.0, torch.ones_like(curr_dist) * 5.0, torch.zeros_like(curr_dist))
            # 到达目标奖励：成功到达给大额奖励
            target_rew[self.agent_success[:, i], i] += 50.0
        
        # 全部成功的团队奖励（所有智能体共享）
        target_rew[self.success_condition, :] += 100.0
        return target_rew

    def _reward_smooth(self):
        """
        平滑奖励：惩罚每个智能体动作的剧烈变化（L2范数）
        
        返回:
            smooth_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        # 计算相邻帧动作差的平方和，值越大表示动作越抖动
        return torch.sum(torch.square(self.actions - self.last_actions), dim=-1)

    def _reward_crash(self):
        """
        坠毁惩罚：当智能体坠毁时给予惩罚
        
        返回:
            crash_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        crash_rew = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.agent_crash] = 1.0
        # 团队惩罚：任何智能体坠毁，所有智能体都受到惩罚
        crash_rew[self.crash_condition, :] += 0.5
        return crash_rew

    def _reward_obstacle(self):
        """
        障碍物惩罚：当智能体接近障碍物时给予惩罚
        
        返回:
            obstacle_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        obstacle_rew = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew
        for i in range(self.num_agents):
            danger_dist = self.obstacle_safe_distance * 0.6
            close_mask = self.min_obstacle_dist[:, i] < danger_dist
            obstacle_rew[close_mask, i] -= (danger_dist - self.min_obstacle_dist[close_mask, i]) / danger_dist
        return obstacle_rew

    def _reward_separation(self):
        """
        分离奖励：惩罚无人机之间距离过近（团队奖励，所有智能体共享）
        
        返回:
            sep_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        sep_rew = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        danger_dist = self.drone_safe_distance * 0.7
        close_mask = self.min_drone_dist < danger_dist
        # 所有智能体共享分离惩罚
        penalty = -(danger_dist - self.min_drone_dist[close_mask]) / danger_dist
        sep_rew[close_mask, :] = penalty.unsqueeze(-1).expand(-1, self.num_agents)
        return sep_rew

    def _reward_progress(self):
        """
        进度奖励：鼓励每个智能体向前飞行并保持稳定姿态
        
        返回:
            progress_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        progress_rew = torch.zeros((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        for i in range(self.num_agents):
            # Y轴前进奖励
            y_progress = self.base_pos[:, i, 1] - self.last_base_pos[:, i, 1]
            progress_rew[:, i] += y_progress * 20.0
            # 高度奖励
            height = self.base_pos[:, i, 2]
            height_good = (height > 0.4) & (height < 1.5)
            progress_rew[:, i] += torch.where(height_good, torch.ones_like(height) * 1.0, -torch.ones_like(height) * 0.5)
            # 姿态稳定奖励
            roll = torch.abs(self.base_euler[:, i, 0])
            pitch = torch.abs(self.base_euler[:, i, 1])
            stable = (roll < 30) & (pitch < 30)
            progress_rew[:, i] += torch.where(stable, torch.ones_like(roll) * 0.5, -torch.ones_like(roll) * 1.0)
        return progress_rew
    
    def _reward_alive(self):
        """
        存活奖励：鼓励每个智能体保持飞行状态，并惩罚原地不动
        
        返回:
            alive_rew: 形状为 (num_envs, num_agents) 的奖励张量
        """
        alive_rew = torch.ones((self.num_envs, self.num_agents), device=gs.device, dtype=gs.tc_float)
        alive_rew[self.agent_crash] = 0
        
        # 惩罚原地不动：如果速度太低且距离目标还远，给予惩罚
        for i in range(self.num_agents):
            speed = torch.norm(self.base_lin_vel[:, i, :], dim=1)
            dist_to_target = torch.norm(self.rel_pos[:, i, :], dim=1)
            # 如果速度小于0.1且距离目标大于1m，惩罚
            lazy_mask = (speed < 0.1) & (dist_to_target > 1.0)
            alive_rew[lazy_mask, i] -= 0.5
        
        return alive_rew

    # ==================== MAPPO特有接口 ====================
    def get_env_info(self):
        """
        获取环境信息（MAPPO算法需要）
        
        返回:
            env_info: 包含环境关键参数的字典
        """
        return {
            "num_agents": self.num_agents,
            "num_obs": self.num_obs,
            "num_state": self.num_state,
            "num_actions": self.num_actions,
            "episode_limit": self.max_episode_length,
        }

    def get_available_actions(self):
        """
        获取可用动作掩码（用于离散动作空间，本环境为连续动作）
        
        返回:
            available_actions_mask: 形状为 (num_envs, num_agents, num_actions) 的掩码
        """
        return self.available_actions_mask
