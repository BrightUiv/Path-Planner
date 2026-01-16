"""
单无人机避障路径规划环境 - PPO版本 + CNN感知

该环境实现了一个单架无人机在有障碍物的3D空间中进行路径规划的强化学习任务。
无人机需要从起点飞行到目标点，同时避开圆柱形障碍物。

观测空间设计（2D平面导航版本）：
1. 自身状态（19维）：
   - 全局位置 (3维)
   - 线速度 (3维)
   - 到目标的距离 (1维)
   - 目标方位角 (1维)
   - 姿态四元数 (4维)
   - 角速度 (3维)
   - 上一步动作 (4维)

2. 障碍物感知（49维）- 线性衰减方案：
   - 7x7x1网格（水平7x7，垂直单层），以无人机为中心
   - 每个格子的值表示该方位障碍物的"危险程度"
   - 使用线性衰减公式：v = max(0, 1 - d / d_max)
   - d=0时v=1（接触），d>=d_max时v=0（无感知）

动作空间：
- 4维连续动作，分别控制4个电机的转速

奖励函数：
- target: 接近目标点的奖励
- progress: Y方向前进、高度保持、姿态稳定的奖励
- alive: 存活奖励
- smooth: 动作平滑惩罚
- crash: 坠毁惩罚
- obstacle: 靠近障碍物的惩罚

依赖：
- Genesis 物理仿真引擎
- PyTorch 深度学习框架
"""
import torch
import math
import copy
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,           # 四元数转欧拉角
    transform_by_quat,     # 通过四元数进行向量变换
    inv_quat,              # 四元数求逆
    transform_quat_by_quat, # 四元数之间的变换
)

# 指定范围内的随机浮点数张量
def gs_rand_float(lower, upper, shape, device):
    """
    生成指定范围内的随机浮点数张量

    Args:
        lower: 随机数下界
        upper: 随机数上界
        shape: 输出张量的形状
        device: 计算设备 (CPU/GPU)

    Returns:
        torch.Tensor: 范围在[lower, upper]之间的随机张量
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class SingleDronePPOEnv:
    """
    单无人机PPO环境 - CNN+MLP混合架构

    该类实现了OpenAI Gym风格的强化学习环境接口，包含：
    - __init__: 初始化环境
    - step: 执行一步仿真
    - reset: 重置环境
    - get_observations: 获取观测值
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        """
        初始化单无人机PPO环境

        Args:
            num_envs (int): 并行环境数量，用于加速训练
            env_cfg (dict): 环境配置，包含：
                - num_actions: 动作维度（4个电机）
                - episode_length_s: 每个episode的最大时长（秒）
                - drone_init_position: 无人机初始位置
                - drone_goal_position: 目标位置
                - obstacle_positions: 障碍物位置列表
                - termination_if_*: 各种终止条件
            obs_cfg (dict): 观测配置，包含：
                - num_state_obs: 自身状态观测维度
                - grid_size: 障碍物感知网格大小
                - obs_scales: 观测缩放因子
            reward_cfg (dict): 奖励配置，包含各奖励项的缩放系数
            command_cfg (dict): 命令配置（目标点相关）
            show_viewer (bool): 是否显示可视化窗口
        """
        
        self.num_envs = num_envs  # 并行环境数量
        self.rendered_env_num = min(5, self.num_envs)  # 可视化时渲染的环境数量（最多5个）

        # 观测空间配置
        self.num_state_obs = obs_cfg["num_state_obs"]  # 自身状态维度（19维）
        # 障碍物网格形状：(X宽度, Y深度, Z高度) = (7, 7, 1)
        # 2D平面导航，水平方向7x7，垂直方向单层
        self.grid_shape = obs_cfg.get("grid_shape", (7, 7, 1))
        self.grid_size_x, self.grid_size_y, self.grid_size_z = self.grid_shape
        self.grid_dim = self.grid_size_x * self.grid_size_y * self.grid_size_z  # 7*7*1=49
        self.grid_resolution = env_cfg.get("grid_resolution", 1.0)  # 每个网格单元的大小(米)

        self.num_privileged_obs = None  # 特权观测（用于教师-学生训练，这里不使用）
        self.num_actions = env_cfg["num_actions"]  # 动作维度：4个电机
        self.num_commands = command_cfg["num_commands"]  # 命令维度：3（目标位置xyz）
        self.device = gs.device  # 计算设备（由Genesis自动选择）

        # 仿真时间步配置
        self.dt = 0.01  # 仿真时间步长（10ms = 100Hz）
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)  # 最大episode步数

        # 保存配置引用
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]  # 观测缩放因子
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])  # 深拷贝奖励缩放系数

        # ==================== 创建仿真场景 ====================
        # Genesis场景是仿真的核心容器，管理所有物理实体和求解器
        if show_viewer:
            # 带可视化的场景配置
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),  # 仿真参数：时间步和子步数
                viewer_options=gs.options.ViewerOptions(
                    max_FPS=env_cfg["max_visualize_FPS"],  # 可视化最大帧率
                    camera_pos=(5.0, 0.0, 5.0),  # 相机位置
                    camera_lookat=(0.0, 0.0, 1.0),  # 相机观察点
                    camera_fov=50,  # 相机视场角
                ),
                vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),  # 渲染的环境索引
                rigid_options=gs.options.RigidOptions(
                    dt=self.dt,
                    constraint_solver=gs.constraint_solver.Newton,  # 约束求解器类型
                    enable_collision=True,  # 启用碰撞检测
                    enable_joint_limit=True,  # 启用关节限制
                ),
                show_viewer=True,
            )
        else:
            # 无可视化的场景配置（用于训练，速度更快）
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

        # 添加地面平面
        self.scene.add_entity(gs.morphs.Plane())

        # ==================== 添加障碍物 ====================
        # 障碍物配置：使用圆柱体作为障碍物
        self.obstacles = []  # 存储障碍物信息的列表
        obstacle_positions = env_cfg.get("obstacle_positions", [])  # 障碍物位置列表
        obstacle_radius = env_cfg.get("obstacle_radius", 0.12)  # 障碍物半径
        obstacle_height = env_cfg.get("obstacle_height", 1.5)  # 障碍物高度

        # 遍历所有障碍物位置，创建圆柱形障碍物
        for pos in obstacle_positions:
            if show_viewer:
                # 可视化模式：创建带表面材质的圆柱体
                obstacle = self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        pos=pos,  # 障碍物位置
                        radius=obstacle_radius,  # 半径
                        height=obstacle_height,  # 高度
                        fixed=True,  # 固定不动
                        collision=True,  # 启用碰撞
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.3, 0.8)),  # 蓝色表面
                    ),
                )
            else:
                # 训练模式：不创建可视化实体（节省计算资源）
                obstacle = None

            # 保存障碍物信息（用于碰撞检测和感知）
            self.obstacles.append({
                "entity": obstacle,  # 障碍物实体引用
                "pos": torch.tensor(pos, device=gs.device),  # 位置张量
                "radius": obstacle_radius,  # 半径
                "height": obstacle_height  # 高度
            })

        # 障碍物距离阈值
        self.obstacle_safe_distance = env_cfg.get("obstacle_safe_distance", 0.4)  # 安全距离（开始惩罚）
        self.obstacle_collision_distance = env_cfg.get("obstacle_collision_distance", 0.18)  # 碰撞距离（终止episode）

        # 障碍物感知参数：线性衰减方案
        # v = max(0, 1 - d / d_max)
        # d_max 是策略开始提前反应的距离，通常取2-4m
        self.perception_d_max = env_cfg.get("perception_d_max", 3.0)  # 感知最大距离(米)

        # 预计算障碍物张量（用于向量化计算，避免每步重复创建）
        if len(self.obstacles) > 0:
            self.obs_positions = torch.stack([obs["pos"] for obs in self.obstacles])  # (num_obs, 3)
            self.obs_radii = torch.tensor([obs["radius"] for obs in self.obstacles],
                                          device=gs.device, dtype=gs.tc_float)  # (num_obs,)
        else:
            self.obs_positions = None
            self.obs_radii = None

       
        # 无人机起点和终点位置
        self.drone_init_position = env_cfg.get("drone_init_position", [0.0, -2.5, 0.6])
        self.drone_goal_position = env_cfg.get("drone_goal_position", [0.0, 2.5, 0.6])

        # 初始姿态四元数（单位四元数，表示无旋转）
        # 四元数格式：[w, x, y, z]
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)  # 逆四元数，用于坐标变换

        # 从URDF文件加载无人机模型（CrazyFlie 2.x四旋翼）
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # 目标点可视化（一个红色小球）
        if env_cfg.get("visualize_target", False):
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj", scale=0.08, pos=self.drone_goal_position,
                    fixed=True, collision=False,  # 固定且不参与碰撞
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.2, 0.2)),  # 红色
                ),
            )

        # ==================== 添加录制相机 ====================
        # 用于录制评估视频
        if env_cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                res=(1280, 720),  # 分辨率
                pos=(5.0, 0.0, 5.0),  # 相机位置
                lookat=(0.0, 0.0, 1.0),  # 观察点
                fov=50,  # 视场角
                GUI=False,  # 不显示GUI
            )
        else:
            self.cam = None

        # 构建场景（初始化所有求解器和并行环境）
        self.scene.build(n_envs=num_envs)

        # ==================== 初始化奖励函数 ====================
        # 动态绑定奖励函数并初始化累积奖励缓冲区
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            # 将奖励缩放系数乘以时间步长（使奖励与时间无关）
            self.reward_scales[name] *= self.dt
            # 动态获取奖励函数（如 _reward_target, _reward_crash 等）
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            # 初始化该奖励项的累积值缓冲区
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ==================== 初始化状态缓冲区 ====================
        # 总观测维度：state(19) + flattened_grid(7*7*1=49) = 68
        self.num_obs = self.num_state_obs + self.grid_dim

        # 核心缓冲区（所有并行环境共享）
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)  # 观测缓冲区
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # 奖励缓冲区
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)  # 重置标志缓冲区
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)  # episode长度缓冲区

        # 目标命令（所有环境共享同一目标点）
        self.command = torch.tensor(self.drone_goal_position, device=gs.device, dtype=gs.tc_float).unsqueeze(0).expand(self.num_envs, -1)

        # 动作缓冲区
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)  # 上一步动作（用于计算动作平滑性）

        # 无人机状态缓冲区
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 位置 [x, y, z]
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)  # 姿态四元数 [w, x, y, z]
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 线速度（机体坐标系）
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 角速度（机体坐标系）
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 欧拉角 [roll, pitch, yaw]
        self.last_base_pos = torch.zeros_like(self.base_pos)  # 上一步位置（用于计算进度）

        # 目标相关状态
        self.rel_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # 相对目标位置
        self.last_rel_pos = torch.zeros_like(self.rel_pos)  # 上一步相对位置
        self.dist_to_target = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # 到目标距离
        self.target_yaw = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # 目标方位角

        # 额外信息字典（用于日志记录和调试）
        self.extras = dict()
        self.extras["observations"] = dict()

    def _compute_obstacle_grid(self):
        """
        计算障碍物感知网格（线性衰减方案）- 完全向量化版本

        该函数实现了一个局部感知系统，采用线性衰减表示障碍物距离：
        - 以无人机为中心，构建一个可配置的3D网格（默认7x7x1用于2D平面导航）
        - 使用线性衰减公式：v = max(0, 1 - d / d_max)

        性能优化：完全使用PyTorch向量化操作，无Python循环，充分利用GPU并行

        Returns:
            torch.Tensor: 形状为 (num_envs, grid_size_x, grid_size_y, grid_size_z) 的网格张量
        """
        # 如果没有障碍物，直接返回空网格
        if len(self.obstacles) == 0:
            return torch.zeros((self.num_envs, self.grid_size_x, self.grid_size_y, self.grid_size_z),
                              device=self.device, dtype=gs.tc_float)

        # 网格参数
        half_x, half_y, half_z = self.grid_size_x // 2, self.grid_size_y // 2, self.grid_size_z // 2
        d_max = self.perception_d_max

        # ==================== 完全向量化计算 ====================
        # 计算相对位置: (num_envs, num_obs, 3)
        rel_pos = self.obs_positions.unsqueeze(0) - self.base_pos.unsqueeze(1)

        # 计算网格坐标: (num_envs, num_obs, 3)
        grid_x = (rel_pos[:, :, 0] / self.grid_resolution + half_x).long()
        grid_y = (rel_pos[:, :, 1] / self.grid_resolution + half_y).long()
        grid_z = (rel_pos[:, :, 2] / self.grid_resolution + half_z).long()

        # 有效性掩码: (num_envs, num_obs)
        valid = ((grid_x >= 0) & (grid_x < self.grid_size_x) &
                 (grid_y >= 0) & (grid_y < self.grid_size_y) &
                 (grid_z >= 0) & (grid_z < self.grid_size_z))

        # 裁剪坐标到有效范围（避免索引越界）
        grid_x = grid_x.clamp(0, self.grid_size_x - 1)
        grid_y = grid_y.clamp(0, self.grid_size_y - 1)
        grid_z = grid_z.clamp(0, self.grid_size_z - 1)

        # 计算感知值: (num_envs, num_obs)
        dist = torch.norm(rel_pos, dim=2) - self.obs_radii.unsqueeze(0)
        perception = torch.clamp(1.0 - torch.clamp(dist, min=0.0) / d_max, min=0.0)
        perception = perception * valid.float()  # 无效位置置零

        # 将3D坐标转换为1D索引，用于scatter_reduce
        # flat_idx = env_idx * (X*Y*Z) + x * (Y*Z) + y * Z + z
        grid_total = self.grid_size_x * self.grid_size_y * self.grid_size_z
        env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)  # (num_envs, 1)
        flat_idx = (env_idx * grid_total +
                    grid_x * (self.grid_size_y * self.grid_size_z) +
                    grid_y * self.grid_size_z +
                    grid_z)  # (num_envs, num_obs)

        # 展平所有数据
        flat_idx = flat_idx.reshape(-1)  # (num_envs * num_obs,)
        perception_flat = perception.reshape(-1)  # (num_envs * num_obs,)

        # 使用scatter_reduce进行最大值聚合
        grid_flat = torch.zeros(self.num_envs * grid_total, device=self.device, dtype=gs.tc_float)
        grid_flat.scatter_reduce_(0, flat_idx, perception_flat, reduce='amax', include_self=True)

        # 重塑为网格形状
        grid = grid_flat.reshape(self.num_envs, self.grid_size_x, self.grid_size_y, self.grid_size_z)

        return grid

    def _compute_target_yaw(self):
        """
        计算目标方位角：从无人机当前朝向到目标方向的角度差

        这个角度帮助无人机判断需要转向多少才能朝向目标。

        计算步骤：
        1. 获取无人机当前的yaw角（绕Z轴旋转角度）
        2. 计算目标点相对于无人机的方向角
        3. 计算两者的差值，即需要调整的角度

        Returns:
            torch.Tensor: 形状为 (num_envs,) 的张量，值范围 [-π, π]
                         正值表示目标在右边，负值表示目标在左边
        """
        # 获取无人机当前朝向（yaw角）
        # base_euler 的单位是度，需要转换为弧度
        current_yaw = self.base_euler[:, 2] * math.pi / 180.0

        # 计算目标方向（只考虑xy平面）
        target_direction = self.rel_pos[:, :2]  # 提取xy分量
        # atan2(y, x) 返回从原点到点(x,y)的角度
        target_yaw = torch.atan2(target_direction[:, 1], target_direction[:, 0])

        # 计算相对角度（目标方向 - 当前朝向）
        relative_yaw = target_yaw - current_yaw

        # 归一化到[-π, π]范围
        # 使用 atan2(sin, cos) 技巧来处理角度环绕
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))

        return relative_yaw

    def step(self, actions):
        """
        执行一步仿真

        这是强化学习环境的核心函数，按以下顺序执行：
        1. 处理并应用动作
        2. 推进物理仿真
        3. 更新状态
        4. 检查终止条件
        5. 计算奖励
        6. 构建观测

        Args:
            actions (torch.Tensor): 形状为 (num_envs, num_actions) 的动作张量
                                   值范围通常是 [-1, 1]，表示电机转速的相对值

        Returns:
            tuple: (obs_buf, rew_buf, reset_buf, extras)
                - obs_buf: 新的观测值
                - rew_buf: 这一步的奖励
                - reset_buf: 哪些环境需要重置
                - extras: 额外信息（包含episode统计等）
        """
        # 裁剪动作到有效范围，防止过大的控制输入
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # 设置无人机电机转速
        # 基础转速 14468.429... RPM 是悬停所需的转速
        # actions * 0.4 表示在基础转速上下浮动 ±40%（降低！使控制更稳定）
        self.drone.set_propellels_rpm((1 + self.actions * 0.4) * 14468.429183500699)

        # 推进物理仿真一个时间步
        self.scene.step()
        self.episode_length_buf += 1  # 增加episode长度计数

        # ==================== 更新无人机状态 ====================
        # 保存上一步的位置（用于计算进度奖励）
        self.last_base_pos[:] = self.base_pos[:]

        # 获取新的位置和姿态
        self.base_pos[:] = self.drone.get_pos()  # 世界坐标系位置
        self.base_quat[:] = self.drone.get_quat()  # 姿态四元数

        # 将四元数转换为欧拉角（roll, pitch, yaw）
        # 使用相对于初始姿态的四元数，然后转换
        self.base_euler[:] = quat_to_xyz(
            transform_quat_by_quat(
                self.inv_base_init_quat.unsqueeze(0).expand(self.num_envs, -1),
                self.base_quat,
            ),
            rpy=True, degrees=True,  # 输出roll-pitch-yaw顺序，单位为度
        )

        # 将速度从世界坐标系转换到机体坐标系
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)  # 线速度
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)  # 角速度

        # 更新目标相关状态
        self.last_rel_pos[:] = self.rel_pos[:]  # 保存上一步的相对位置
        self.rel_pos = self.command - self.base_pos  # 计算新的相对位置
        self.dist_to_target = torch.norm(self.rel_pos, dim=1)  # 到目标的欧氏距离
        self.target_yaw = self._compute_target_yaw()  # 目标方位角

        # 计算最近障碍物距离（用于终止条件和奖励）
        self.min_obstacle_dist = self._get_min_obstacle_distance()

        # ==================== 终止条件检查 ====================
        # 坠毁条件：姿态过大、高度过低、或碰撞障碍物
        crash = (
            # pitch角过大（向前/后倾斜超限）
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            # roll角过大（向左/右倾斜超限）
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            # 高度过低（接近地面）
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
            # 碰撞障碍物
            | (self.min_obstacle_dist < self.obstacle_collision_distance)
        )

        # 成功条件：到达目标点附近
        success = self.dist_to_target < self.env_cfg["at_target_threshold"]

        # 保存条件标志（用于奖励计算）
        self.crash_condition = crash
        self.success_condition = success

        # 重置标志：超时、坠毁或成功都需要重置
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | crash | success

        # 重置需要重置的环境
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ==================== 计算奖励 ====================
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            # 计算每个奖励项，乘以对应的缩放系数
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew  # 累积用于日志记录

        # ==================== 构建观测 ====================
        # 自身状态观测（19维）
        state_obs = torch.cat([
            self.base_pos * self.obs_scales["pos"],  # 全局位置 (3维)
            torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),  # 线速度 (3维)，裁剪到[-1,1]
            (self.dist_to_target * self.obs_scales["dist"]).unsqueeze(-1),  # 到目标距离 (1维)
            (self.target_yaw * self.obs_scales["yaw"]).unsqueeze(-1),  # 目标方位角 (1维)
            self.base_quat,  # 姿态四元数 (4维)
            torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),  # 角速度 (3维)，裁剪到[-1,1]
            self.last_actions,  # 上一步动作 (4维)
        ], dim=-1)  # 总共19维

        # 障碍物网格观测：grid_shape -> 展平（默认7x7x1=49维）
        obstacle_grid = self._compute_obstacle_grid()
        obstacle_grid_flat = obstacle_grid.reshape(self.num_envs, -1)  # (num_envs, grid_dim)

        # 拼接完整观测：19 + grid_dim（默认19+49=68维）
        self.obs_buf = torch.cat([state_obs, obstacle_grid_flat], dim=-1)

        # 更新上一步动作
        self.last_actions[:] = self.actions[:]

        # 设置critic观测（在这个实现中与actor观测相同）
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _get_min_obstacle_distance(self):
        """
        计算每个环境中无人机与最近障碍物的距离 - 向量化版本

        只考虑XY平面的距离（因为障碍物是垂直的圆柱体）。
        距离计算为无人机到圆柱体表面的距离（减去圆柱体半径）。

        Returns:
            torch.Tensor: 形状为 (num_envs,) 的距离张量
        """
        if len(self.obstacles) == 0:
            return torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float) * 100.0

        # 向量化计算：所有环境到所有障碍物的距离
        # drone_pos_xy: (num_envs, 2) -> (num_envs, 1, 2)
        # obs_positions_xy: (num_obs, 2) -> (1, num_obs, 2)
        drone_pos_xy = self.base_pos[:, :2].unsqueeze(1)  # (num_envs, 1, 2)
        obs_pos_xy = self.obs_positions[:, :2].unsqueeze(0)  # (1, num_obs, 2)

        # 计算到所有障碍物表面的距离: (num_envs, num_obs)
        dist_to_surface = torch.norm(drone_pos_xy - obs_pos_xy, dim=2) - self.obs_radii.unsqueeze(0)

        # 取每个环境的最小距离: (num_envs,)
        min_dist = dist_to_surface.min(dim=1)[0]

        return min_dist

    def get_observations(self):
        """
        获取当前观测值

        Returns:
            tuple: (obs_buf, extras)
                - obs_buf: 观测张量
                - extras: 包含critic观测等额外信息
        """
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    # 特权观测
    def get_privileged_observations(self):
        """
        获取特权观测（用于教师-学生训练范式）

        在这个实现中不使用特权观测，所以返回None。

        Returns:
            None
        """
        return None
    
    # 重置指定的环境到初始状态
    def reset_idx(self, envs_idx):
        """
        重置指定的环境到初始状态

        Args:
            envs_idx (torch.Tensor): 需要重置的环境索引列表
        """
        if len(envs_idx) == 0:
            return

        # 设置初始位置
        init_pos = torch.tensor(self.drone_init_position, device=gs.device)
        self.base_pos[envs_idx] = init_pos
        self.last_base_pos[envs_idx] = init_pos
        self.base_quat[envs_idx] = self.base_init_quat

        # 重置无人机物理状态
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.zero_all_dofs_velocity(envs_idx)  # 清零所有自由度的速度

        # 重置状态缓冲区
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # 记录episode统计信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # 计算平均奖励率（每秒奖励）
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            # 重置累积奖励
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        """
        重置所有环境

        通常在训练开始时调用。

        Returns:
            tuple: (obs_buf, None)
                - obs_buf: 初始观测值
                - None: 占位符（与step返回格式一致）
        """
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None
    
    # 奖励函数的设计：以下是各个奖励项的实现，它们在每一步被调用并累加

    def _reward_target(self):
        """
        目标接近奖励

        这是最主要的奖励项，鼓励无人机接近目标点。

        奖励组成：
        1. 距离减少奖励：鼓励每一步减少与目标的距离
        2. 距离惩罚：离目标越远惩罚越大
        3. 接近奖励：距离小于2m时额外奖励
        4. 非常接近奖励：距离小于1m时更大的额外奖励
        5. 成功奖励：到达目标时的大额奖励

        Returns:
            torch.Tensor: 形状为 (num_envs,) 的奖励张量
        """
        # 计算距离变化量
        last_dist = torch.norm(self.last_rel_pos, dim=1)
        dist_reduction = last_dist - self.dist_to_target  # 正值表示在接近目标

        # 基础奖励：距离减少量 * 10
        target_rew = dist_reduction * 10.0

        # 距离惩罚：鼓励保持较近距离
        target_rew -= self.dist_to_target * 0.1

        # 接近奖励阶梯：
        # 距离 < 2m 时额外奖励 2.0
        target_rew += torch.where(self.dist_to_target < 2.0,
                                  torch.ones_like(self.dist_to_target) * 2.0,
                                  torch.zeros_like(self.dist_to_target))
        # 距离 < 1m 时额外奖励 5.0
        target_rew += torch.where(self.dist_to_target < 1.0,
                                  torch.ones_like(self.dist_to_target) * 5.0,
                                  torch.zeros_like(self.dist_to_target))

        # 成功到达目标的大额奖励
        target_rew[self.success_condition] += 100.0

        return target_rew

    def _reward_smooth(self):
        """
        动作平滑性惩罚

        惩罚动作的剧烈变化，鼓励平滑的控制输入。
        这有助于：
        1. 减少机械磨损
        2. 提高飞行稳定性
        3. 节省能源

        Returns:
            torch.Tensor: 动作变化的平方和（负值乘以缩放系数后成为惩罚）
        """
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_crash(self):
        """
        坠毁惩罚

        当无人机坠毁（姿态失控、高度过低、碰撞障碍物）时给予惩罚。

        Returns:
            torch.Tensor: 坠毁环境返回1.0，其他返回0.0
                         （乘以负的缩放系数后成为惩罚）
        """
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    def _reward_obstacle(self):
        """
        障碍物接近惩罚

        当无人机靠近障碍物时给予梯度惩罚，距离越近惩罚越大。
        这个软惩罚有助于无人机学会保持安全距离，而不仅仅是避免碰撞。

        Returns:
            torch.Tensor: 障碍物惩罚值（0到-1之间）
        """
        obstacle_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        if len(self.obstacles) == 0:
            return obstacle_rew

        # 危险距离阈值（安全距离的60%）
        danger_dist = self.obstacle_safe_distance * 0.6

        # 找出进入危险区域的环境
        close_mask = self.min_obstacle_dist < danger_dist

        # 计算惩罚：距离越近惩罚越大（线性插值）
        obstacle_rew[close_mask] = -(danger_dist - self.min_obstacle_dist[close_mask]) / danger_dist

        return obstacle_rew

    # 进度方向的奖励
    def _reward_progress(self):
        """
        综合奖励（2D平面导航优化版）

        综合奖励，包含多个方面：
        1. Y方向前进（主要移动方向）
        2. 高度保持在0.5-0.7m之间（强化约束，保持在同一平面）
        3. 姿态稳定性

        Returns:
            torch.Tensor: 综合进度奖励
        """
        progress_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # Y方向进度（从起点到终点的主要方向）
        y_progress = self.base_pos[:, 1] - self.last_base_pos[:, 1]
        progress_rew += y_progress * 20.0  # Y方向前进奖励

        # 高度保持奖励（2D平面导航关键）：强制保持在0.5-0.7m之间
        height = self.base_pos[:, 2]
        # 理想高度范围：0.5-0.7m（中心0.6m）
        height_good = (height > 0.5) & (height < 0.7)
        progress_rew += torch.where(height_good,
                                    torch.ones_like(height) * 2.0,   # 高度理想：大奖励
                                    -torch.ones_like(height) * 2.0)  # 高度偏离：大惩罚

        # 额外惩罚：高度偏离过大（超出0.4-0.8m范围）
        height_bad = (height < 0.4) | (height > 0.8)
        progress_rew += torch.where(height_bad,
                                    -torch.ones_like(height) * 3.0,  # 严重偏离：额外惩罚
                                    torch.zeros_like(height))

        # 姿态稳定奖励：roll和pitch都在30度以内
        roll = torch.abs(self.base_euler[:, 0])
        pitch = torch.abs(self.base_euler[:, 1])
        stable = (roll < 30) & (pitch < 30)
        progress_rew += torch.where(stable,
                                    torch.ones_like(roll) * 0.5,   # 姿态稳定：奖励
                                    -torch.ones_like(roll) * 1.0)  # 姿态不稳：惩罚

        return progress_rew
    
    # 存活奖励
    def _reward_alive(self):
        """
        存活奖励

        简单的存活奖励，只要没有坠毁就给予正奖励。
        这有助于鼓励无人机保持飞行，不要冒险坠毁。

        Returns:
            torch.Tensor: 存活环境返回1.0，坠毁环境返回0.0
        """
        alive_rew = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float)
        alive_rew[self.crash_condition] = 0  # 坠毁的环境没有存活奖励
        return alive_rew

    # 姿态稳定性的奖励
    def _reward_stability(self):
        """
        奖励无人机保持稳定的姿态和低角速度。
        这是学习飞行的基础，必须先学会稳定悬停。

        奖励组成：
        1. 角速度惩罚：旋转越快惩罚越大
        2. 姿态奖励：roll/pitch越小奖励越大
        3. 水平速度惩罚：防止失控漂移

        Returns:
            torch.Tensor: 稳定性奖励值
        """
        stability_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # 1. 角速度惩罚（核心！防止旋转失控）
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
        stability_rew -= ang_vel_norm * 0.5  # 角速度越大，惩罚越大

        # 2. 姿态奖励：roll和pitch越接近0越好
        roll = torch.abs(self.base_euler[:, 0])  # 度
        pitch = torch.abs(self.base_euler[:, 1])  # 度

        # 姿态良好（roll和pitch都小于15度）给予奖励
        attitude_good = (roll < 15) & (pitch < 15)
        stability_rew += torch.where(attitude_good,
                                     torch.ones_like(roll) * 2.0,   # 姿态好：奖励
                                     torch.zeros_like(roll))

        # 姿态一般（15-30度）给予小奖励
        attitude_ok = (roll < 30) & (pitch < 30) & ~attitude_good
        stability_rew += torch.where(attitude_ok,
                                     torch.ones_like(roll) * 0.5,
                                     torch.zeros_like(roll))

        # 3. 水平速度适度惩罚（防止失控漂移）
        horizontal_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        stability_rew -= torch.clamp(horizontal_vel - 1.0, min=0.0) * 0.2  # 超过1m/s才惩罚

        return stability_rew
