"""
单无人机避障路径规划训练脚本 - PPO版本 + CNN-MLP混合架构
"""
import argparse
import os
import pickle
import shutil
from importlib import metadata

# 检查rsl-rl-lib版本
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from single_drone_ppo_env import SingleDronePPOEnv
from cnn_mlp_actor_critic import CNNMLPActorCritic


def get_train_cfg(exp_name, max_iterations):
    """
    获取PPO训练配置
    """
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",  # 使用默认类作为占位符
            "activation": "elu",
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "init_noise_std": 0.5,
        },
        # 自定义网络参数（用于后续创建）
        "cnn_mlp_policy": {
            "cnn_channels": [16, 32, 64],
            "mlp_hidden_dims": [128, 128],
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 100,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def get_cfgs():
    """
    获取环境配置
    """
    env_cfg = {
        "num_actions": 4,  # 4个电机
        "termination_if_roll_greater_than": 80,
        "termination_if_pitch_greater_than": 80,
        "termination_if_close_to_ground": 0.02,
        "drone_init_position": [0.0, -2.5, 0.8],
        "drone_goal_position": [0.0, 2.5, 0.8],
        "episode_length_s": 30.0,
        "at_target_threshold": 0.4,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        # 障碍物配置
        "obstacle_positions": [
            [-0.5, -1.5, 1.0], [0.5, -1.5, 1.0],
            [-1.0, -0.5, 1.0], [0.0, -0.5, 1.0], [1.0, -0.5, 1.0],
            [-0.5, 0.5, 1.0], [0.5, 0.5, 1.0],
            [-1.0, 1.5, 1.0], [0.0, 1.5, 1.0], [1.0, 1.5, 1.0],
        ],
        "obstacle_radius": 0.1,
        "obstacle_height": 2.0,
        "obstacle_safe_distance": 0.3,
        "obstacle_collision_distance": 0.12,
        # 障碍物网格感知配置
        "grid_resolution": 0.5,  # 每个网格单元的大小（米）
    }

    # 观测配置
    # 自身状态：位置(3) + 速度(3) + 到目标距离(1) + 目标方位角(1) + 四元数(4) + 角速度(3) + 上一动作(4) = 19维
    # 障碍物网格：3x3x3 = 27维
    # 总观测维度：19 + 27 = 46维
    obs_cfg = {
        "num_state_obs": 19,  # 自身状态维度
        "grid_size": 3,  # 障碍物感知网格大小 3x3x3
        "obs_scales": {
            "pos": 1 / 5.0,  # 位置缩放
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
            "dist": 1 / 5.0,  # 距离缩放
            "yaw": 1 / 3.14159,  # 角度缩放
        },
    }

    reward_cfg = {
        "reward_scales": {
            "target": 50.0,
            "progress": 30.0,
            "alive": 5.0,
            "smooth": -1e-6,
            "crash": -5.0,
            "obstacle": -1.0,
        },
    }

    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-cnn-ppo")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    # 初始化 Genesis
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 128)
        print(f"[Visualization Mode] Reduced num_envs to {args.num_envs}")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = SingleDronePPOEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    print(f"\n{'='*60}")
    print(f"Single Drone PPO Training with CNN-MLP Architecture")
    print(f"{'='*60}")
    print(f"Environments: {args.num_envs}")
    print(f"State obs dim: {env.num_state_obs}")
    print(f"Total obs dim: {env.num_obs} (state={env.num_state_obs} + grid={obs_cfg['grid_size']**3})")
    print(f"Obstacle grid: {obs_cfg['grid_size']}x{obs_cfg['grid_size']}x{obs_cfg['grid_size']}")
    print(f"Actions: {env.num_actions}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"{'='*60}\n")

    # 首先创建OnPolicyRunner（会使用默认ActorCritic）
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # 创建CNN-MLP混合网络
    policy_cfg = train_cfg["policy"]
    cnn_mlp_cfg = train_cfg["cnn_mlp_policy"]
    actor_critic = CNNMLPActorCritic(
        num_state_obs=env.num_state_obs,
        num_actions=env.num_actions,
        grid_size=obs_cfg["grid_size"],
        cnn_channels=cnn_mlp_cfg["cnn_channels"],
        mlp_hidden_dims=cnn_mlp_cfg["mlp_hidden_dims"],
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],
        activation=policy_cfg["activation"],
        init_noise_std=policy_cfg["init_noise_std"],
    ).to(gs.device)

    # 替换runner中的默认网络
    runner.alg.actor_critic = actor_critic
    print(f"Replaced default ActorCritic with CNNMLPActorCritic\n")

    # 开始训练
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()


"""
# 单无人机PPO训练命令

# 无可视化（快速训练）
python single_drone_ppo_train.py -e single-drone-cnn-ppo -B 4096 --max_iterations 1000

# 带可视化
python single_drone_ppo_train.py -e single-drone-cnn-ppo -B 64 --max_iterations 1000 -v
"""
