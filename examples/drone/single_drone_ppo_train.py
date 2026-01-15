"""
单无人机避障路径规划训练脚本 - PPO版本 + CNN-MLP混合架构

该脚本使用Proximal Policy Optimization (PPO)算法训练无人机避障策略。
PPO是一种高效稳定的策略梯度算法，通过限制策略更新的幅度来保证训练稳定性。

训练流程：
1. 初始化Genesis物理引擎
2. 创建并行环境（数千个环境同时运行）
3. 创建CNN-MLP混合网络
4. 使用rsl_rl的OnPolicyRunner进行训练
5. 定期保存模型检查点

关键配置：
- env_cfg: 环境配置（障碍物、终止条件等）
- obs_cfg: 观测配置（状态维度、网格大小等）
- reward_cfg: 奖励配置（各奖励项的权重）
- train_cfg: 训练配置（PPO超参数、学习率等）

依赖：
- Genesis: 物理仿真引擎
- rsl-rl-lib==2.2.4: 强化学习训练库
- PyTorch: 深度学习框架

使用方法：
# 无可视化（快速训练）
python single_drone_ppo_train.py -e single-drone-cnn-ppo -B 4096 --max_iterations 1000

# 带可视化（调试用）
python single_drone_ppo_train.py -e single-drone-cnn-ppo -B 64 --max_iterations 1000 -v
"""
import argparse
import os
import pickle
import shutil
from importlib import metadata

# ==================== 版本检查 ====================
# 确保安装了正确版本的rsl-rl库
# 注意：rsl_rl（带下划线）和rsl-rl-lib是不同的包
try:
    try:
        # 如果安装了旧的rsl_rl包，抛出错误
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        # 检查rsl-rl-lib的版本
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

# 导入训练相关模块
from rsl_rl.runners import OnPolicyRunner  # PPO训练器
import genesis as gs
from single_drone_ppo_env import SingleDronePPOEnv  # 自定义环境
from cnn_mlp_actor_critic import CNNMLPActorCritic  # 自定义网络


def get_train_cfg(exp_name, max_iterations):
    """
    获取PPO训练配置

    该函数返回一个包含所有PPO训练超参数的字典。
    这些参数会被rsl_rl的OnPolicyRunner使用。

    Args:
        exp_name (str): 实验名称，用于日志目录命名
        max_iterations (int): 最大训练迭代次数

    Returns:
        dict: 训练配置字典，包含以下主要部分：
            - algorithm: PPO算法超参数
            - policy: 策略网络配置
            - cnn_mlp_policy: CNN-MLP混合网络配置
            - runner: 训练运行器配置
    """
    train_cfg_dict = {
        # ==================== PPO算法超参数 ====================
        "algorithm": {
            "class_name": "PPO",  # 算法类名

            # PPO核心参数
            "clip_param": 0.2,  # PPO裁剪参数ε，限制策略更新幅度
                                # 新旧策略比率被裁剪到[1-ε, 1+ε]范围内

            "desired_kl": 0.01,  # 目标KL散度，用于自适应学习率调整
                                 # 如果KL散度超过这个值，降低学习率

            "entropy_coef": 0.01,  # 熵正则化系数，鼓励探索
                                   # 值越大，策略越倾向于随机

            "gamma": 0.99,  # 折扣因子γ，决定未来奖励的重要性
                           # 0.99表示长期奖励很重要

            "lam": 0.95,  # GAE(广义优势估计)的λ参数
                          # 用于平衡偏差和方差的权衡

            "learning_rate": 0.0003,  # 初始学习率
                                       # 使用自适应调整

            "max_grad_norm": 1.0,  # 梯度裁剪阈值，防止梯度爆炸

            "num_learning_epochs": 5,  # 每批数据的训练轮数
                                       # PPO可以在同一批数据上多次更新

            "num_mini_batches": 4,  # 每批数据分成的小批次数
                                    # 用于mini-batch SGD

            "schedule": "adaptive",  # 学习率调整策略
                                     # "adaptive"根据KL散度调整

            "use_clipped_value_loss": True,  # 是否裁剪价值函数损失
                                              # 与策略裁剪类似，提高稳定性

            "value_loss_coef": 1.0,  # 价值函数损失的权重
        },

        "init_member_classes": {},  # 初始化成员类（高级用法）

        # ==================== 策略网络配置 ====================
        "policy": {
            "class_name": "ActorCritic",  # 默认类名（作为占位符）
                                          # 实际使用我们自定义的CNNMLPActorCritic

            "activation": "elu",  # 激活函数类型
                                  # ELU比ReLU更平滑

            "actor_hidden_dims": [256, 128],  # Actor网络隐藏层维度
            "critic_hidden_dims": [256, 128],  # Critic网络隐藏层维度

            "init_noise_std": 0.5,  # 初始动作噪声标准差
                                    # 较小的值使初始策略更保守
        },

        # ==================== CNN-MLP混合网络配置 ====================
        # 这些参数用于创建我们自定义的网络
        "cnn_mlp_policy": {
            "cnn_channels": [32, 64],  # CNN各层的通道数
                                       # 两层卷积 + 自适应池化

            "mlp_hidden_dims": [128, 128],  # 状态MLP的隐藏层维度
        },

        # ==================== 训练运行器配置 ====================
        "runner": {
            "checkpoint": -1,  # 检查点迭代号（-1表示不加载）
            "experiment_name": exp_name,  # 实验名称
            "load_run": -1,  # 加载的运行编号（-1表示不加载）
            "log_interval": 1,  # 日志打印间隔（每多少次迭代打印一次）
            "max_iterations": max_iterations,  # 最大迭代次数
            "record_interval": -1,  # 视频录制间隔（-1表示不录制）
            "resume": False,  # 是否从检查点恢复
            "resume_path": None,  # 恢复路径
            "run_name": "",  # 运行名称（为空则自动生成）
        },

        "runner_class_name": "OnPolicyRunner",  # 运行器类名

        # ==================== 数据收集配置 ====================
        "num_steps_per_env": 100,  # 每个环境每次迭代收集的步数
                                   # 总步数 = num_envs * num_steps_per_env

        "save_interval": 100,  # 模型保存间隔（每多少次迭代保存一次）

        "empirical_normalization": None,  # 经验归一化（None表示不使用）

        "seed": 1,  # 随机种子，用于可重复性
    }
    return train_cfg_dict


'''
    整个仿真环境之中的配置信息
'''
def get_cfgs():
    """
    获取环境配置

    该函数返回环境、观测、奖励和命令的配置字典。
    这些配置决定了仿真环境的行为和学习目标。

    Returns:
        tuple: (env_cfg, obs_cfg, reward_cfg, command_cfg)
    """
    # ==================== 环境配置 ====================
    env_cfg = {
        # 动作空间
        "num_actions": 4,  # 4个电机的转速控制

        # 终止条件（这些条件会导致episode提前结束）
        "termination_if_roll_greater_than": 80,  # roll角超过80度
        "termination_if_pitch_greater_than": 80,  # pitch角超过80度
        "termination_if_close_to_ground": 0.02,  # 高度低于2cm

        # 起点和终点位置
        "drone_init_position": [0.0, -2.5, 0.8],  # 无人机初始位置 (x, y, z)
        "drone_goal_position": [0.0, 2.5, 0.8],   # 目标位置
        # 任务：从y=-2.5飞到y=2.5，需要穿越障碍物区域

        # Episode配置
        "episode_length_s": 30.0,  # 每个episode最大时长（秒）
        "at_target_threshold": 0.4,  # 到达目标的距离阈值（米）

        # 动作配置
        "simulate_action_latency": True,  # 模拟动作延迟
        "clip_actions": 1.0,  # 动作裁剪范围 [-1, 1]

        # 可视化配置
        "visualize_target": False,  # 是否可视化目标点
        "visualize_camera": False,  # 是否使用录制相机
        "max_visualize_FPS": 60,  # 可视化最大帧率

        # ==================== 障碍物配置 ====================
        # 10个圆柱形障碍物，形成需要穿越的障碍物阵列
        "obstacle_positions": [
            # 第一排（y=-1.5）
            [-0.5, -1.5, 1.0], [0.5, -1.5, 1.0],
            # 第二排（y=-0.5）
            [-1.0, -0.5, 1.0], [0.0, -0.5, 1.0], [1.0, -0.5, 1.0],
            # 第三排（y=0.5）
            [-0.5, 0.5, 1.0], [0.5, 0.5, 1.0],
            # 第四排（y=1.5）
            [-1.0, 1.5, 1.0], [0.0, 1.5, 1.0], [1.0, 1.5, 1.0],
        ],
        "obstacle_radius": 0.1,  # 障碍物半径（米）
        "obstacle_height": 2.0,  # 障碍物高度（米）
        "obstacle_safe_distance": 0.3,  # 安全距离（开始惩罚的距离）
        "obstacle_collision_distance": 0.12,  # 碰撞距离（终止episode的距离）

        # 障碍物网格感知配置
        "grid_resolution": 0.5,  # 每个网格单元的大小（米）
                                 # 3x3x3网格覆盖 1.5m x 1.5m x 1.5m 的空间
    }

    # ==================== 观测配置 ====================
    # 总观测维度 = 自身状态(19) + 障碍物网格(7*7*3=147) = 166维
    obs_cfg = {
        # 自身状态观测维度分解：
        # - 位置: 3维 (x, y, z)
        # - 速度: 3维 (vx, vy, vz)
        # - 到目标距离: 1维
        # - 目标方位角: 1维
        # - 姿态四元数: 4维 (w, x, y, z)
        # - 角速度: 3维 (wx, wy, wz)
        # - 上一步动作: 4维 (4个电机)
        # 总计: 19维
        "num_state_obs": 19,

        # 障碍物感知网格形状 (X, Y, Z) = (7, 7, 3)
        # 水平方向更宽（7x7），垂直方向较窄（3）
        # 适合圆柱形障碍物的避障任务
        "grid_shape": (7, 7, 3),  # 7*7*3 = 147个格子

        # 观测缩放因子（归一化）
        "obs_scales": {
            "pos": 1 / 5.0,  # 位置缩放：5米范围映射到[-1, 1]
            "lin_vel": 1 / 3.0,  # 速度缩放：3m/s映射到[-1, 1]
            "ang_vel": 1 / 3.14159,  # 角速度缩放：π rad/s映射到[-1, 1]
            "dist": 1 / 5.0,  # 距离缩放：5米映射到[0, 1]
            "yaw": 1 / 3.14159,  # 角度缩放：π弧度映射到[-1, 1]
        },
    }

    '''
        奖励函数的设计：
    '''
    # 奖励函数 = Σ (reward_scale * reward_function)
    # 正值表示奖励，负值表示惩罚
    reward_cfg = {
        "reward_scales": {
            # 目标接近奖励（主要奖励）
            "target": 50.0,  # 接近目标的奖励权重

            # 进度奖励（辅助奖励）
            "progress": 30.0,  # Y方向前进、高度保持、姿态稳定

            # 存活奖励
            "alive": 5.0,  # 每步存活的基础奖励

            # 惩罚项（负权重）
            "smooth": -1e-6,  # 动作平滑性惩罚（很小的值，主要用于打破平局）
            "crash": -5.0,  # 坠毁惩罚
            "obstacle": -1.0,  # 接近障碍物的惩罚
        },
    }

    # ==================== 命令配置 ====================
    command_cfg = {
        "num_commands": 3  # 命令维度：目标位置 (x, y, z)
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    """
    主函数：执行PPO训练流程

    训练流程：
    1. 解析命令行参数
    2. 初始化Genesis物理引擎
    3. 创建日志目录
    4. 创建环境和网络
    5. 执行训练
    """
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-cnn-ppo",
                        help="实验名称，用于日志目录命名")
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                        help="是否启用可视化（会降低训练速度）")
    parser.add_argument("-B", "--num_envs", type=int, default=4096,
                        help="并行环境数量（越多越快，但需要更多GPU内存）")
    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="最大训练迭代次数")
    args = parser.parse_args()

    # ==================== 初始化Genesis ====================
    # backend=gs.gpu: 使用GPU加速
    # precision="32": 使用32位浮点数
    # logging_level="warning": 只显示警告级别以上的日志
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # ==================== 日志目录设置 ====================
    log_dir = f"logs/{args.exp_name}"

    # 获取配置
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # 清理旧的日志目录（重新开始训练）
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 可视化模式：减少环境数量以提高渲染性能
    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 128)  # 限制最多128个环境
        print(f"[Visualization Mode] Reduced num_envs to {args.num_envs}")

    # 保存配置（用于后续评估和复现）
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # 创建仿真环境
    env = SingleDronePPOEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    # ==================== 打印训练信息 ====================
    grid_shape = obs_cfg["grid_shape"]
    print(f"\n{'='*60}")
    print(f"Single Drone PPO Training with CNN-MLP Architecture")
    print(f"{'='*60}")
    print(f"Environments: {args.num_envs}")
    print(f"State obs dim: {env.num_state_obs}")
    print(f"Total obs dim: {env.num_obs} (state={env.num_state_obs} + grid={grid_shape[0]*grid_shape[1]*grid_shape[2]})")
    print(f"Obstacle grid: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}")
    print(f"Actions: {env.num_actions}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"{'='*60}\n")

    # ==================== 创建训练器 ====================
    # OnPolicyRunner是rsl_rl提供的PPO训练器
    # 它会创建一个默认的ActorCritic网络，我们稍后会替换它
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # ==================== 创建CNN-MLP混合网络 ====================
    # 使用我们自定义的网络架构替换默认网络
    policy_cfg = train_cfg["policy"]
    cnn_mlp_cfg = train_cfg["cnn_mlp_policy"]

    actor_critic = CNNMLPActorCritic(
        num_state_obs=env.num_state_obs,  # 自身状态维度: 19
        num_actions=env.num_actions,  # 动作维度: 4
        grid_size=obs_cfg["grid_shape"],  # 网格形状: (7, 7, 3)
        cnn_channels=cnn_mlp_cfg["cnn_channels"],  # CNN通道: [32, 64]
        mlp_hidden_dims=cnn_mlp_cfg["mlp_hidden_dims"],  # MLP隐藏层: [128, 128]
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],  # Actor隐藏层: [256, 128]
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],  # Critic隐藏层: [256, 128]
        activation=policy_cfg["activation"],  # 激活函数: elu
        init_noise_std=policy_cfg["init_noise_std"],  # 初始噪声: 0.5
    ).to(gs.device)  # 将网络移到GPU

    # 替换runner中的默认网络
    # 这是使用自定义网络的关键步骤
    runner.alg.actor_critic = actor_critic
    print(f"Replaced default ActorCritic with CNNMLPActorCritic\n")

    # ==================== 开始训练 ====================
    # learn()方法执行完整的PPO训练循环：
    # 1. 收集轨迹（使用当前策略与环境交互）
    # 2. 计算优势函数
    # 3. PPO策略更新
    # 4. 记录日志
    # 5. 保存检查点
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,  # 随机初始化episode长度，避免所有环境同步重置
    )


if __name__ == "__main__":
    main()


"""
# 单无人机PPO训练命令示例

# 无可视化（快速训练，推荐用于正式训练）
python single_drone_ppo_train.py -e single-drone-cnn-ppo -B 4096 --max_iterations 1000

# 带可视化（用于调试和观察学习过程）
python single_drone_ppo_train.py -e single-drone-cnn-ppo -B 64 --max_iterations 1000 -v

# 参数说明：
# -e, --exp_name: 实验名称，日志将保存到 logs/<exp_name>/
# -B, --num_envs: 并行环境数量
#   - 4096: 适合训练，需要约8GB显存
#   - 64-128: 适合可视化调试
# --max_iterations: 训练迭代次数
#   - 每次迭代收集 num_envs * num_steps_per_env 步数据
#   - 1000次迭代约收集4亿步数据（4096*100*1000）
# -v, --vis: 启用可视化窗口

# 训练过程中会定期保存检查点到 logs/<exp_name>/
# 包括：
# - model_<iter>.pt: 模型权重
# - cfgs.pkl: 配置文件（用于评估时加载）
"""
