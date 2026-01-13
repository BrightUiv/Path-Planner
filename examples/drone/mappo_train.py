"""
多无人机避障路径规划训练脚本 - MAPPO版本
支持可视化训练过程，类似PPO训练体验

MAPPO核心特点：
- 集中式训练分布式执行(CTDE)架构
- Actor使用局部观测，Critic使用全局状态
- 多智能体共享策略网络
"""
import argparse
import os
import pickle
import shutil
import torch
import time
from collections import deque

import genesis as gs
from mappo_env import MultiDroneMAPPOEnv
from mappo_algorithm import MAPPO, MAPPOBuffer



"""
返回:
    env_cfg: 环境配置（无人机数量、终止条件、障碍物等）
    obs_cfg: 观测配置（观测维度、缩放系数）
    reward_cfg: 奖励配置（各奖励项权重）
    command_cfg: 命令配置（目标位置维度）
"""
def get_cfgs():
    num_drones = 3  # 无人机数量
    
    # ==================== 环境配置 ====================
    env_cfg = {
        "num_drones": num_drones,           # 无人机数量
        "num_actions": 4,                   # 单架无人机动作维度（4个螺旋桨）
        # 终止条件
        "termination_if_roll_greater_than": 80,    # roll角度超过80度终止
        "termination_if_pitch_greater_than": 80,   # pitch角度超过80度终止
        "termination_if_close_to_ground": 0.02,    # 高度低于0.02m终止
        # 无人机初始位置（Y轴负方向）
        "drone_init_positions": [
            [-1.0, -2.5, 0.8],
            [0.0, -2.5, 0.8],
            [1.0, -2.5, 0.8],
        ],
        # 无人机目标位置（Y轴正方向）
        "drone_goal_positions": [
            [-1.0, 2.5, 0.8],
            [0.0, 2.5, 0.8],
            [1.0, 2.5, 0.8],
        ],
        "episode_length_s": 35.0,           # 单个episode回合的最大时长（秒）
        "at_target_threshold": 0.5,         # 到达目标的距离阈值
        "simulate_action_latency": True,    # 是否模拟动作延迟
        "clip_actions": 1.0,                # 动作裁剪范围
        "visualize_target": False,          # 是否可视化目标点
        "visualize_camera": False,          # 是否启用录制相机
        "max_visualize_FPS": 60,            # 可视化最大帧率
        # 障碍物配置（10个圆柱体，分布在飞行路径上）
        "obstacle_positions": [
            [-0.5, -1.5, 1.0], [0.5, -1.5, 1.0],
            [-1.0, -0.5, 1.0], [0.0, -0.5, 1.0], [1.0, -0.5, 1.0],
            [-0.5, 0.5, 1.0], [0.5, 0.5, 1.0],
            [-1.0, 1.5, 1.0], [0.0, 1.5, 1.0], [1.0, 1.5, 1.0],
        ],
        "obstacle_radius": 0.1,             # 障碍物半径
        "obstacle_height": 2.0,             # 障碍物高度
        "obstacle_safe_distance": 0.3,      # 障碍物安全距离（开始惩罚）
        "obstacle_collision_distance": 0.12, # 障碍物碰撞距离（终止）
        "drone_safe_distance": 0.35,        # 无人机间安全距离
        "drone_collision_distance": 0.2,    # 无人机间碰撞距离（终止）
    }
    
    # ==================== 观测配置 ====================
    obs_cfg = {
        "num_obs_per_drone": 20,  # 单架无人机观测维度：智能体ID(3)+相对位置(3)+四元数(4)+线速度(3)+角速度(3)+动作(4)
        "obs_scales": {           # 观测缩放系数（归一化用）
            "rel_pos": 1 / 3.0,   # 相对位置缩放（3m范围映射到[-1,1]）
            "lin_vel": 1 / 3.0,   # 线速度缩放
            "ang_vel": 1 / 3.14159,  # 角速度缩放（除以π）
        },
    }
    
    # ==================== 奖励配置（密集引导信号 + 存活奖励）====================
    reward_cfg = {
        "reward_scales": {
            "target": 50.0,       # 目标奖励（含密集距离信号）
            "progress": 30.0,     # 前进+高度+姿态奖励
            "alive": 5.0,         # 存活奖励（鼓励保持飞行）
            "smooth": -1e-6,      # 极小平滑惩罚（避免动作抖动）
            "crash": -5.0,        # 坠机惩罚
            "obstacle": -1.0,     # 避障惩罚
            "separation": -0.5,   # 无人机间距惩罚
        },
    }
    
    # ==================== 命令配置 ====================
    command_cfg = {"num_commands": 3}  # 目标位置xyz坐标
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg


"""
获取MAPPO算法配置

返回:
    mappo_cfg: MAPPO超参数字典
"""
def get_mappo_cfg():
    return {
        "lr_actor": 3e-4,       # Actor学习率（适中，避免不稳定）
        "lr_critic": 5e-4,      # Critic学习率
        "gamma": 0.99,          # 折扣因子
        "lam": 0.95,            # GAE lambda
        "clip_param": 0.2,      # PPO裁剪参数
        "entropy_coef": 0.005,  # 降低熵系数，减少随机性
        "value_loss_coef": 0.5, # 价值损失系数
        "max_grad_norm": 0.5,   # 更严格的梯度裁剪，防止崩溃
        "num_epochs": 5,        # 增加epoch数，充分利用数据
        "batch_size": 4096,     # 4090显存大，用更大batch
        "share_actor": False,   # 独立Actor
    }


"""
MAPPO训练Runner - 支持可视化

核心职责：
1. 管理环境交互：控制多个并行环境同时运行
2. 经验收集：将(观测,动作,奖励)存入缓冲区
3. 网络更新：调用MAPPO算法更新Actor和Critic
4. 训练监控：记录奖励、成功率等统计信息
5. 模型管理：定期保存/加载模型检查点
"""
class MAPPORunner:
    
    def __init__(self, env, mappo_cfg, log_dir, device, num_steps_per_env=100):
        """
        初始化MAPPO训练器
        
        参数:
            env: 多无人机环境实例
            mappo_cfg: MAPPO算法配置
            log_dir: 日志保存目录
            device: 计算设备（CPU/GPU）
            num_steps_per_env: 每次迭代每个环境收集的步数
        """
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.num_steps_per_env = num_steps_per_env
        
        # 从环境获取维度信息
        num_agents = env.num_drones              # 智能体数量
        obs_dim = env.num_obs                    # 局部观测维度（23维）
        global_obs_dim = env.num_state           # 全局状态维度（69维）
        action_dim = env.num_actions             # 动作维度（4维）
        
        # 初始化MAPPO算法（包含Actor和Critic网络）
        self.mappo = MAPPO(
            num_agents=num_agents,           # 智能体数量（3架无人机）
            obs_dim=obs_dim,                 # 单个智能体的观测维度
            global_obs_dim=global_obs_dim,   # 全局状态维度（Critic输入）
            action_dim=action_dim,           # 单个智能体的动作维度
            device=device,
            **mappo_cfg                      # 展开超参数配置
        )
        
        # 初始化经验缓冲区（存储训练数据）
        self.buffer = MAPPOBuffer(
            num_envs=env.num_envs,           # 并行环境数量
            num_agents=num_agents,           # 智能体数量
            num_steps=num_steps_per_env,     # 每次迭代收集的步数
            obs_dim=obs_dim,                 # 观测维度
            global_obs_dim=global_obs_dim,   # 全局状态维度
            action_dim=action_dim,           # 动作维度
            device=device
        )
        
        # 训练统计（滑动窗口）
        self.reward_history = deque(maxlen=100)   # 最近100次迭代的奖励
        self.success_history = deque(maxlen=100)  # 最近100次迭代的成功率
        self.total_timesteps = 0                  # 总交互步数
        self.start_time = None                    # 训练开始时间
    
    """
    训练主循环
    
    参数:
        max_iterations: 最大迭代次数
        save_interval: 模型保存间隔
        log_interval: 日志输出间隔
    """
    def learn(self, max_iterations, save_interval=100, log_interval=1):
        self.start_time = time.time()
        obs_buf, _ = self.env.reset()
        
        # 初始化局部观测（reset后需要构建）
        self._update_observations()
        
        # 打印训练信息
        print("\n" + "="*60)
        print("MAPPO Multi-Drone Training Started")
        print(f"Num Envs: {self.env.num_envs} | Num Drones: {self.env.num_drones}")
        print(f"Obs Dim (per drone): {self.env.num_obs} | Global Obs Dim: {self.env.num_state}")
        print("="*60 + "\n")
        
        for iteration in range(max_iterations):
            iter_start = time.time()
            episode_rewards = []
            episode_successes = []
            
            # ==================== 收集经验 ====================
            # 在每个环境中执行num_steps_per_env步，收集训练数据
            for step in range(self.num_steps_per_env):
                # 获取局部观测（Actor输入）和全局状态（Critic输入）
                # MAPPO核心：Actor只看局部信息，Critic看全局信息
                local_obs = self.env.obs_buf.clone()     # (num_envs, num_agents, obs_dim)
                global_obs = self.env.state_buf.clone()  # (num_envs, global_obs_dim)
                
                # 采样动作和计算价值（推理模式，不计算梯度）
                with torch.no_grad():
                    actions, log_probs = self.mappo.get_actions(local_obs)  # Actor输出动作
                    values = self.mappo.get_value(global_obs)               # Critic估计价值
                
                # 环境交互：执行动作，获取下一状态、奖励、终止标志
                obs_buf, state_buf, rew_buf, reset_buf, extras = self.env.step(actions)
                
                # 存储经验到缓冲区（用于后续网络更新）
                # 奖励取所有智能体的平均值（共享奖励）
                rewards = rew_buf.mean(dim=-1)
                self.buffer.store(
                    obs=local_obs,           # 局部观测
                    global_obs=global_obs,   # 全局状态
                    actions=actions,         # 执行的动作
                    log_probs=log_probs,     # 动作的对数概率（PPO需要）
                    rewards=rewards,         # 获得的奖励
                    dones=reset_buf.float(), # 是否终止
                    values=values            # 价值估计
                )
                
                # 记录奖励统计
                episode_rewards.append(rewards.mean().item())
                
                # 统计成功率（当有episode结束时）
                if "episode" in extras:
                    episode_successes.append(self.env.success_condition.float().mean().item())
                
                # 累计总交互步数
                self.total_timesteps += self.env.num_envs
            
            # ==================== 网络更新 ====================
            # 计算最后一步的价值（用于GAE优势估计的bootstrap）
            with torch.no_grad():
                last_global_obs = self.env.state_buf.clone()
                last_value = self.mappo.get_value(last_global_obs)
            
            # GAE(Generalized Advantage Estimation)：计算优势函数
            # 优势 = 实际回报 - 价值估计，用于指导策略更新方向
            self.buffer.compute_gae(last_value.detach(), self.mappo.gamma, self.mappo.lam)
            
            # 使用收集的经验更新Actor和Critic网络
            losses = self.mappo.update(self.buffer)
            
            # ==================== 统计和日志 ====================
            # 计算本次迭代的平均奖励并加入历史记录
            mean_reward = sum(episode_rewards) / len(episode_rewards)
            self.reward_history.append(mean_reward)
            
            # 记录成功率
            if episode_successes:
                mean_success = sum(episode_successes) / len(episode_successes)
                self.success_history.append(mean_success)
            
            # 计算训练速度（每秒处理的环境步数）
            iter_time = time.time() - iter_start
            fps = (self.num_steps_per_env * self.env.num_envs) / iter_time
            
            # 日志输出
            if (iteration + 1) % log_interval == 0:
                elapsed = time.time() - self.start_time
                avg_reward = sum(self.reward_history) / len(self.reward_history)
                avg_success = sum(self.success_history) / len(self.success_history) if self.success_history else 0
                
                print(f"Iter {iteration+1:4d}/{max_iterations} | "
                      f"Reward: {mean_reward:7.3f} (avg: {avg_reward:7.3f}) | "
                      f"Success: {avg_success*100:5.1f}% | "
                      f"FPS: {fps:6.0f} | "
                      f"Time: {elapsed/60:5.1f}min")
                
                # 每10次迭代输出详细损失
                if (iteration + 1) % 10 == 0:
                    print(f"  └─ Actor Loss: {losses['actor_loss']:.4f} | "
                          f"Critic Loss: {losses['critic_loss']:.4f} | "
                          f"Entropy: {losses['entropy']:.4f}")
            
            # 保存模型
            if (iteration + 1) % save_interval == 0:
                self.save(os.path.join(self.log_dir, f"mappo_model_{iteration+1}.pt"))
        
        # 最终保存
        self.save(os.path.join(self.log_dir, "mappo_model_final.pt"))
        
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print(f"Training Complete! Total Time: {total_time/60:.1f} min")
        print(f"Final Avg Reward: {sum(self.reward_history)/len(self.reward_history):.3f}")
        print("="*60)
    

    """
        更新观测（用于reset后初始化）
        
        构建每个智能体的局部观测和全局状态
    """

    def _update_observations(self):
        """更新观测（用于reset后初始化，调用环境的计算方法）"""
        self.env._compute_observations()
    
    def save(self, path):
        """保存模型"""
        self.mappo.save(path)
        print(f"  [Saved] {path}")
    
    def load(self, path):
        """加载模型"""
        self.mappo.load(path)
        print(f"  [Loaded] {path}")
    
    def get_inference_policy(self):
        """获取推理策略（用于评估）"""
        return self.mappo



def main():
    """
    主函数：程序入口
    
    执行流程：
    1. 解析命令行参数
    2. 初始化Genesis仿真引擎
    3. 加载环境和算法配置
    4. 创建多无人机环境
    5. 创建MAPPO训练器
    6. 启动训练循环
    """
    
    # ==================== 命令行参数 ====================
    parser = argparse.ArgumentParser(description="MAPPO多无人机训练脚本")
    parser.add_argument("-e", "--exp_name", type=str, default="multi-drone-mappo",
                        help="实验名称（用于日志目录）")
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                        help="启用可视化窗口")
    parser.add_argument("-B", "--num_envs", type=int, default=8192,
                        help="并行环境数量（4090推荐8192）")
    parser.add_argument("--max_iterations", type=int, default=800,
                        help="最大训练迭代次数")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="模型保存间隔")
    parser.add_argument("--num_steps", type=int, default=64,
                        help="每次迭代每个环境的步数（更短更频繁更新）")
    args = parser.parse_args()

    # ==================== 初始化Genesis仿真引擎 ====================
    # Genesis是一个GPU加速的物理仿真引擎，用于无人机动力学模拟
    gs.init(
        backend=gs.gpu,           # 使用GPU后端加速仿真
        precision="32",           # 32位浮点精度（平衡速度和精度）
        logging_level="warning",  # 只显示警告级别以上的日志
        performance_mode=True     # 启用性能优化模式
    )

    # ==================== 加载配置 ====================
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    mappo_cfg = get_mappo_cfg()

    # 清理并创建日志目录
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ==================== 可视化配置 ====================
    if args.vis:
        env_cfg["visualize_target"] = True
        args.num_envs = min(args.num_envs, 128)  # 可视化时减少环境数量以保证帧率
        print(f"[Visualization Mode] Reduced num_envs to {args.num_envs}")
    else:
        print(f"[Training Mode] Using {args.num_envs} parallel environments")

    # ==================== 保存配置（用于评估时加载）====================
    # 将所有配置序列化保存，确保评估时使用相同的配置
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, mappo_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # ==================== 创建环境 ====================
    # MultiDroneMAPPOEnv封装了Genesis仿真，提供强化学习标准接口
    env = MultiDroneMAPPOEnv(
        num_envs=args.num_envs,      # 并行环境数量（GPU并行加速）
        env_cfg=env_cfg,             # 环境配置
        obs_cfg=obs_cfg,             # 观测配置
        reward_cfg=reward_cfg,       # 奖励配置
        command_cfg=command_cfg,     # 命令配置
        show_viewer=args.vis,        # 是否显示可视化窗口
    )

    # ==================== 创建训练器并开始训练 ====================
    # MAPPORunner负责整个训练流程的管理
    runner = MAPPORunner(
        env,                              # 环境实例
        mappo_cfg,                        # MAPPO算法配置
        log_dir,                          # 日志保存目录
        device=gs.device,                 # 计算设备（自动选择GPU/CPU）
        num_steps_per_env=args.num_steps  # 每次迭代每个环境收集的步数
    )
    
    # 启动训练主循环
    runner.learn(
        max_iterations=args.max_iterations,   # 最大迭代次数
        save_interval=args.save_interval      # 模型保存间隔
    )


if __name__ == "__main__":
    main()


"""
==================== MAPPO训练命令示例 ====================

# 1. 无可视化快速训练（推荐）
python mappo_train.py -e multi-drone-mappo -B 4096 --max_iterations 800

# 2. 带可视化观察训练过程
python mappo_train.py -e multi-drone-mappo -B 64 --max_iterations 800 -v

# 3. 自定义参数训练
python mappo_train.py -e my-exp -B 2048 --max_iterations 500 --save_interval 50 --num_steps 128

==================== 参数说明 ====================
-e, --exp_name      : 实验名称，日志保存在 logs/<exp_name>/
-v, --vis           : 启用可视化窗口（会自动减少环境数量）
-B, --num_envs      : 并行环境数量（越多训练越快，但需要更多显存）
--max_iterations    : 最大训练迭代次数
--save_interval     : 模型保存间隔（每N次迭代保存一次）
--num_steps         : 每次迭代每个环境收集的步数

==================== 输出文件 ====================
logs/<exp_name>/
├── cfgs.pkl                    # 配置文件（评估时加载）
├── mappo_model_100.pt          # 中间检查点
├── mappo_model_200.pt
├── ...
└── mappo_model_final.pt        # 最终模型
"""
