"""
单无人机避障路径规划评估脚本 - PPO版本 + CNN-MLP
支持可视化评估和视频录制
"""
import argparse
import os
import pickle
import torch

import genesis as gs
from single_drone_ppo_env import SingleDronePPOEnv
from cnn_mlp_actor_critic import CNNMLPActorCritic

# 检查rsl-rl-lib版本
from importlib import metadata
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="single-drone-cnn-ppo")
    parser.add_argument("--ckpt", type=int, default=-1, help="检查点迭代数，-1表示加载最新模型")
    parser.add_argument("--record", action="store_true", default=False, help="录制视频")
    parser.add_argument("--episodes", type=int, default=5, help="评估的episode数量")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"

    # 加载配置
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found at {cfg_path}")
        return

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))

    # 评估时不计算奖励
    reward_cfg["reward_scales"] = {}

    # 可视化配置
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    env = SingleDronePPOEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # 创建CNN-MLP网络
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

    # 确定模型路径
    if args.ckpt > 0:
        model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        # 查找最新的模型
        model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            print(f"Error: No model found in {log_dir}")
            return
        # 按迭代数排序，取最大的
        model_files.sort(key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
        model_path = os.path.join(log_dir, model_files[-1])

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        available = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if available:
            print(f"Available models: {available}")
        return

    # 加载模型
    checkpoint = torch.load(model_path, map_location=gs.device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    print(f"Loaded model from {model_path}")

    max_steps_per_episode = int(env_cfg["episode_length_s"] / env.dt)

    print("\n" + "="*50)
    print("PPO Single-Drone Evaluation (CNN-MLP)")
    print(f"Episodes: {args.episodes} | Max Steps: {max_steps_per_episode}")
    print("="*50 + "\n")

    with torch.no_grad():
        if args.record:
            # 录制模式
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.join(script_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"ppo_{args.exp_name}.mp4")

            print(f"Recording to {video_path}...")
            env.cam.start_recording()

            for ep in range(args.episodes):
                obs, _ = env.reset()

                for step in range(max_steps_per_episode):
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    env.cam.render()

                    if dones.any():
                        print(f"  Episode {ep+1}: Done at step {step}")
                        break

            env.cam.stop_recording(save_to_filename=video_path, fps=60)
            print(f"\nVideo saved to {video_path}")

        else:
            # 交互式评估
            success_count = 0

            for ep in range(args.episodes):
                obs, _ = env.reset()
                episode_reward = 0

                for step in range(max_steps_per_episode):
                    actions = actor_critic.act_inference(obs)
                    obs, rews, dones, infos = env.step(actions)
                    episode_reward += rews.sum().item()

                    if dones.any():
                        if hasattr(env, 'success_condition') and env.success_condition.any():
                            success_count += 1
                            print(f"Episode {ep+1}: SUCCESS (step {step}, reward {episode_reward:.2f})")
                        else:
                            print(f"Episode {ep+1}: Done (step {step}, reward {episode_reward:.2f})")
                        break
                else:
                    print(f"Episode {ep+1}: Timeout (reward {episode_reward:.2f})")

            print(f"\nCompleted: {args.episodes} episodes | Success rate: {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()
