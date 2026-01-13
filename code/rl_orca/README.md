# RL + VO/ORCA（单机无人机动态避障）最小原型

这个目录给出一个“RL 调参 + VO/ORCA 生成速度指令”的最小可运行骨架：

- `orca.py`：简化版 VO/ORCA（在速度空间做安全修正，输出期望速度）。
- `env_genesis.py`：Genesis 场景环境（1 架无人机 + 若干动态障碍物），把 ORCA 的速度指令转成位置小步目标，再用 PID 输出电机转速。
- `policy.py`：A2C 风格 actor-critic（连续动作），动作用于调 ORCA 的关键参数（预测时域/安全裕度/避障权重/最大速度缩放）。
- `train_rl_orca.py`：训练脚本（不依赖外部 RL 库，仅 PyTorch）。
- `run_orca_baseline.py`：不训练，固定参数跑 ORCA baseline。

运行示例（从仓库根目录）：

- ORCA 基线：`python code/rl_orca/run_orca_baseline.py --backend cpu --vis`
- RL 训练：`python code/rl_orca/train_rl_orca.py --backend cpu --total_updates 200`
- RL 可视化评估：`python code/rl_orca/train_rl_orca.py --backend cpu --vis --total_updates 1 --eval_only --ckpt path/to.pt`

说明：
- 这里的 ORCA 实现是“VO/ORCA 风格”的简化版（用于演示“RL 学参数 + VO/ORCA 产出速度”这一创新点）；如果你需要严格 ORCA（线性约束 + 线性规划求最接近 preferred velocity 的可行解），可以在此基础上替换 `orca.py` 的求解器。

