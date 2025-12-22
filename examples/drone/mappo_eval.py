"""
多无人机避障路径规划评估脚本 - MAPPO版本
支持可视化评估和视频录制

使用方法:
    # 基本评估（可视化）
    python mappo_eval.py -e multi-drone-mappo
    
    # 指定检查点
    python mappo_eval.py -e multi-drone-mappo --ckpt 500
    
    # 录制视频
    python mappo_eval.py -e multi-drone-mappo --record
"""
import argparse
import os
import pickle
import torch

import genesis as gs
from mappo_env import MultiDroneMAPPOEnv
from mappo_algorithm import MAPPO


def main():
    parser = argparse.ArgumentParser(description="MAPPO多无人机评估脚本")
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo",
                        help="实验名称（与训练时一致）")
    parser.add_argument("--ckpt", type=int, default=-1, 
                        help="检查点迭代数，-1表示加载最终模型")
    parser.add_argument("--record", action="store_true", default=False, 
                        help="录制视频")
    parser.add_argument("--episodes", type=int, default=5, 
                        help="评估的episode数量")
    args = parser.parse_args()

    # 初始化Genesis
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    
    # ==================== 加载训练时保存的配置 ====================
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"Error: 配置文件不存在 {cfg_path}")
        print("请确认实验名称正确，或先运行训练脚本")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, mappo_cfg = pickle.load(open(cfg_path, "rb"))
    
    # 评估时可以不计算奖励（可选）
    # reward_cfg["reward_scales"] = {}

    # 可视化配置
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    # ==================== 创建环境 ====================
    env = MultiDroneMAPPOEnv(
        num_envs=1,  # 评估时只用1个环境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # ==================== 创建MAPPO并加载模型 ====================
    mappo = MAPPO(
        num_agents=env.num_drones,
        obs_dim=env.num_obs,
        global_obs_dim=env.num_state,
        action_dim=env.num_actions,
        device=gs.device,
        **mappo_cfg
    )
    
    # 确定模型路径
    if args.ckpt > 0:
        model_path = os.path.join(log_dir, f"mappo_model_{args.ckpt}.pt")
    else:
        model_path = os.path.join(log_dir, "mappo_model_final.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: 模型文件不存在 {model_path}")
        # 列出可用的模型
        available = [f for f in os.listdir(log_dir) if f.startswith("mappo_model_") and f.endswith(".pt")]
        if available:
            print(f"可用的模型: {available}")
        return
    
    # 加载模型权重
    mappo.load(model_path)
    print(f"已加载模型: {model_path}")

    max_steps_per_episode = int(env_cfg["episode_length_s"] / env.dt)
    
    print("\n" + "="*60)
    print("MAPPO Multi-Drone Evaluation")
    print(f"Num Drones: {env.num_drones} | Episodes: {args.episodes}")
    print(f"Max Steps per Episode: {max_steps_per_episode}")
    print("="*60 + "\n")
    
    # ==================== 评估循环 ====================
    with torch.no_grad():
        if args.record:
            # 录制模式
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_dir = os.path.join(script_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"mappo_{args.exp_name}.mp4")
            
            print(f"录制视频到 {video_path}...")
            env.cam.start_recording()
            
            for ep in range(args.episodes):
                obs, _ = env.reset()
                env._compute_observations()
                
                for step in range(max_steps_per_episode):
                    # 获取局部观测
                    local_obs = env.obs_buf.clone()
                    
                    # 使用训练好的策略选择动作（确定性模式）
                    actions, _ = mappo.get_actions(local_obs, deterministic=True)
                    
                    # 执行动作
                    obs, state, rews, dones, infos = env.step(actions)
                    env.cam.render()
                    
                    if dones.any():
                        print(f"  Episode {ep+1}: Done at step {step}")
                        break
            
            env.cam.stop_recording(save_to_filename=video_path, fps=60)
            print(f"\n视频已保存到 {video_path}")
        
        else:
            # 交互式评估
            success_count = 0
            total_rewards = []
            
            for ep in range(args.episodes):
                obs, _ = env.reset()
                env._compute_observations()
                episode_reward = 0
                
                for step in range(max_steps_per_episode):
                    # 获取局部观测
                    local_obs = env.obs_buf.clone()
                    
                    # 使用训练好的策略选择动作（确定性模式）
                    actions, _ = mappo.get_actions(local_obs, deterministic=True)
                    
                    # 执行动作
                    obs, state, rews, dones, infos = env.step(actions)
                    episode_reward += rews.mean().item()
                    
                    if dones.any():
                        if hasattr(env, 'success_condition') and env.success_condition.any():
                            success_count += 1
                            print(f"Episode {ep+1}: ✓ SUCCESS (step {step}, reward {episode_reward:.2f})")
                        else:
                            print(f"Episode {ep+1}: ✗ Failed (step {step}, reward {episode_reward:.2f})")
                        break
                else:
                    print(f"Episode {ep+1}: ⏱ Timeout (reward {episode_reward:.2f})")
                
                total_rewards.append(episode_reward)
            
            # 打印统计结果
            print("\n" + "="*60)
            print("评估结果")
            print(f"成功率: {success_count}/{args.episodes} ({success_count/args.episodes*100:.1f}%)")
            print(f"平均奖励: {sum(total_rewards)/len(total_rewards):.2f}")
            print("="*60)


if __name__ == "__main__":
    main()
