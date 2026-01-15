"""
单无人机避障路径规划评估脚本 - PPO版本 + CNN-MLP

该脚本用于评估训练好的无人机避障策略，支持：
1. 交互式可视化评估：实时观察无人机的飞行行为
2. 视频录制模式：将评估过程录制为MP4视频

评估流程：
1. 加载训练时保存的配置文件
2. 创建环境（单个环境，启用可视化）
3. 加载训练好的模型权重
4. 使用确定性策略（无探索噪声）进行评估
5. 统计成功率和奖励

使用方法：
# 交互式评估（观察5个episode）
python single_drone_ppo_eval.py -e single-drone-cnn-ppo --episodes 5

# 录制评估视频
python single_drone_ppo_eval.py -e single-drone-cnn-ppo --record

# 使用特定检查点
python single_drone_ppo_eval.py -e single-drone-cnn-ppo --ckpt 500

依赖：
- Genesis: 物理仿真引擎
- rsl-rl-lib==2.2.4: 强化学习训练库
- PyTorch: 深度学习框架
"""
import argparse
import os
import pickle
import torch

import genesis as gs
from single_drone_ppo_env import SingleDronePPOEnv  # 自定义环境
from cnn_mlp_actor_critic import CNNMLPActorCritic  # 自定义网络

# ==================== 版本检查 ====================
# 确保安装了正确版本的rsl-rl库
from importlib import metadata
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


def main():
    """
    主函数：执行模型评估

    评估流程：
    1. 解析命令行参数
    2. 初始化Genesis物理引擎
    3. 加载训练配置
    4. 创建环境和网络
    5. 加载模型权重
    6. 运行评估循环
    """
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-cnn-ppo",
                        help="实验名称，需要与训练时使用的名称一致")
    parser.add_argument("--ckpt", type=int, default=-1,
                        help="检查点迭代数，-1表示自动加载最新模型")
    parser.add_argument("--record", action="store_true", default=False,
                        help="是否录制评估视频")
    parser.add_argument("--episodes", type=int, default=5,
                        help="评估的episode数量")
    args = parser.parse_args()

    # ==================== 初始化Genesis ====================
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # 日志目录（与训练时相同）
    log_dir = f"logs/{args.exp_name}"

    # ==================== 加载训练配置 ====================
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        print("请确保已经运行过训练脚本，或者检查实验名称是否正确")
        return

    # 加载配置文件（包含环境、观测、奖励、命令和训练配置）
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))

    # ==================== 评估模式配置 ====================
    # 评估时不计算奖励（避免影响性能和日志）
    reward_cfg["reward_scales"] = {}

    # 启用可视化
    env_cfg["visualize_target"] = True  # 显示目标点
    env_cfg["visualize_camera"] = args.record  # 如果需要录制，启用录制相机
    env_cfg["max_visualize_FPS"] = 60  # 可视化帧率

    # ==================== 创建环境 ====================
    # 评估时只需要单个环境
    env = SingleDronePPOEnv(
        num_envs=1,  # 单个环境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,  # 显示可视化窗口
    )

    # ==================== 创建网络 ====================
    # 使用与训练时相同的网络架构
    policy_cfg = train_cfg["policy"]
    cnn_mlp_cfg = train_cfg["cnn_mlp_policy"]

    actor_critic = CNNMLPActorCritic(
        num_state_obs=env.num_state_obs,  # 自身状态维度: 19
        num_actions=env.num_actions,  # 动作维度: 4
        grid_size=obs_cfg["grid_size"],  # 网格大小: 3
        cnn_channels=cnn_mlp_cfg["cnn_channels"],  # CNN通道
        mlp_hidden_dims=cnn_mlp_cfg["mlp_hidden_dims"],  # MLP隐藏层
        actor_hidden_dims=policy_cfg["actor_hidden_dims"],  # Actor隐藏层
        critic_hidden_dims=policy_cfg["critic_hidden_dims"],  # Critic隐藏层
        activation=policy_cfg["activation"],  # 激活函数
        init_noise_std=policy_cfg["init_noise_std"],  # 初始噪声（评估时不使用）
    ).to(gs.device)

    # ==================== 确定模型路径 ====================
    if args.ckpt > 0:
        # 使用指定的检查点
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        # 自动查找最新的模型
        model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            print(f"Error: No model found in {log_dir}")
            print("请先运行训练脚本生成模型")
            return
        # 按迭代数排序，取最大的（最新的）
        model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
        model_path = os.path.join(log_dir, model_files[-1])

    # 验证模型文件存在
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        available = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if available:
            print(f"Available models: {available}")
        return

    # ==================== 加载模型权重 ====================
    checkpoint = torch.load(model_path, map_location=gs.device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=False)
    actor_critic.eval()  # 设置为评估模式（关闭dropout等）
    print(f"Loaded model from {model_path}")

    # 计算每个episode的最大步数
    max_steps_per_episode = int(env_cfg["episode_length_s"] / env.dt)

    # ==================== 打印评估信息 ====================
    print("\n" + "="*50)
    print("PPO Single-Drone Evaluation (CNN-MLP)")
    print(f"Episodes: {args.episodes} | Max Steps: {max_steps_per_episode}")
    print("="*50 + "\n")

    # ==================== 评估循环 ====================
    # 使用torch.no_grad()禁用梯度计算，节省内存和计算
    with torch.no_grad():
        if args.record:
            # ==================== 录制模式 ====================
            # 创建视频输出目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.join(script_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"ppo_{args.exp_name}.mp4")

            print(f"Recording to {video_path}...")
            env.cam.start_recording()  # 开始录制

            # 运行评估episode并录制
            for ep in range(args.episodes):
                obs, _ = env.reset()  # 重置环境

                for step in range(max_steps_per_episode):
                    # 使用确定性策略（无噪声）
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    env.cam.render()  # 渲染当前帧

                    if dones.any():
                        print(f"  Episode {ep+1}: Done at step {step}")
                        break

            # 停止录制并保存视频
            env.cam.stop_recording(save_to_filename=video_path, fps=60)
            print(f"\nVideo saved to {video_path}")

        else:
            # ==================== 交互式评估模式 ====================
            success_count = 0  # 成功到达目标的次数

            for ep in range(args.episodes):
                obs, _ = env.reset()  # 重置环境
                episode_reward = 0  # 累积奖励（虽然评估时reward_scales为空，但保留统计）

                for step in range(max_steps_per_episode):
                    # 使用确定性策略（无噪声）
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    episode_reward += rews.sum().item()

                    if dones.any():
                        # 检查是否成功到达目标
                        if hasattr(env, 'success_condition') and env.success_condition.any():
                            success_count += 1
                            print(f"Episode {ep+1}: SUCCESS (step {step}, reward {episode_reward:.2f})")
                        else:
                            # 未成功（可能是超时、坠毁或碰撞）
                            print(f"Episode {ep+1}: Done (step {step}, reward {episode_reward:.2f})")
                        break
                else:
                    # 达到最大步数（超时）
                    print(f"Episode {ep+1}: Timeout (reward {episode_reward:.2f})")

            # ==================== 打印评估统计 ====================
            print(f"\nCompleted: {args.episodes} episodes | Success rate: {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()
