"""
多无人机避障路径规划训练脚本 - PPO版本
将多架无人机视为一个整体，使用标准PPO算法训练
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
from multi_drone_ppo_env import MultiDronePPOEnv


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
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 0.5,
            "class_name": "ActorCritic",
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
    num_drones = 3
    
    env_cfg = {
        "num_drones": num_drones,
        "num_actions": 4,  # 每架无人机4个电机
        "termination_if_roll_greater_than": 80,
        "termination_if_pitch_greater_than": 80,
        "termination_if_close_to_ground": 0.02,
        "drone_init_positions": [
            [-1.0, -2.5, 0.8],
            [0.0, -2.5, 0.8],
            [1.0, -2.5, 0.8],
        ],
        "drone_goal_positions": [
            [-1.0, 2.5, 0.8],
            [0.0, 2.5, 0.8],
            [1.0, 2.5, 0.8],
        ],
        "episode_length_s": 35.0,
        "at_target_threshold": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
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
        "drone_safe_distance": 0.35,
        "drone_collision_distance": 0.2,
    }
    
    # PPO版本：观测维度 = 每架无人机观测 * 无人机数量 + 其他无人机相对位置
    # 每架无人机基础观测: 17维 (rel_pos:3 + quat:4 + lin_vel:3 + ang_vel:3 + last_action:4)
    # 加上其他无人机的相对位置: (num_drones-1) * 3
    obs_per_drone = 17 + (num_drones - 1) * 3  # 17 + 6 = 23
    
    obs_cfg = {
        "num_obs": obs_per_drone * num_drones,  # 23 * 3 = 69
        "num_obs_per_drone": obs_per_drone,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
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
            "separation": -0.5,
        },
    }
    
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-ppo")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=800)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

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

    env = MultiDronePPOEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# PPO多无人机训练命令

# 无可视化（快速训练）
python multi_drone_ppo_train.py -e multi-drone-ppo -B 4096 --max_iterations 800

# 带可视化
python multi_drone_ppo_train.py -e multi-drone-ppo -B 64 --max_iterations 800 -v
"""
