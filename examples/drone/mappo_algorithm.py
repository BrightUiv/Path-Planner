"""
- Actor: 每个智能体根据局部观测独立决策
- Critic: 使用全局状态信息评估联合价值函数
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class MAPPOActor(nn.Module):
    """
    分布式Actor网络 - 每个智能体独立决策
    输入: 单个智能体的局部观测
    输出: 动作的均值和标准差（高斯策略）
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256], init_std=0.5):
        super().__init__()
        
        self.action_dim = action_dim
        
        # 构建MLP骨干网络
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),  # ELU激活函数，比ReLU更平滑
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)  # 特征提取网络
        self.mean_head = nn.Linear(prev_dim, action_dim)  # 输出动作均值
        # 可学习的log标准差，初始值较大以鼓励早期探索
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))
        
        # 正交初始化：有助于训练稳定性
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # 输出层使用较小的增益，使初始输出接近零
        nn.init.orthogonal_(self.mean_head.weight, gain=0.1)
        nn.init.zeros_(self.mean_head.bias)
    
    def forward(self, obs):
        """前向传播，输出动作分布参数"""
        features = self.backbone(obs)
        # tanh将均值限制在[-1, 1]范围内
        mean = torch.tanh(self.mean_head(features))
        # 标准差限制在[0.1, 1.0]，防止过小（确定性）或过大（随机）
        std = torch.clamp(self.log_std.exp(), min=0.1, max=1.0).expand_as(mean)
        return mean, std
    
    def get_action(self, obs, deterministic=False):
        """
        采样动作
        deterministic=True: 评估时直接使用均值
        deterministic=False: 训练时从高斯分布采样
        """
        mean, std = self.forward(obs)
        if deterministic:
            return mean, torch.zeros(obs.shape[0], device=obs.device)
        
        dist = Normal(mean, std)  # 构建高斯分布
        action = torch.clamp(dist.sample(), -1.0, 1.0)  # 采样并裁剪
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算对数概率
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """评估给定动作的对数概率和熵（用于PPO更新）"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)  # 熵用于鼓励探索
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    集中式Critic网络 - 使用全局状态评估价值
    输入: 所有智能体的联合观测（全局状态）
    输出: 状态价值V(s)
    """
    
    def __init__(self, global_obs_dim, hidden_dims=[512, 512, 256]):
        super().__init__()
        
        # 构建价值网络
        layers = []
        prev_dim = global_obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出单一价值
        
        self.network = nn.Sequential(*layers)
        
        # 正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, global_obs):
        """输出状态价值"""
        return self.network(global_obs).squeeze(-1)


class MAPPOBuffer:
    """
    MAPPO经验回放缓冲区
    存储轨迹数据用于PPO更新
    """
    
    def __init__(self, num_envs, num_agents, num_steps, obs_dim, global_obs_dim, action_dim, device):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.device = device
        self.ptr = 0  # 当前写入位置
        
        # 预分配存储空间
        self.obs = torch.zeros((num_steps, num_envs, num_agents, obs_dim), device=device)  # 局部观测
        self.global_obs = torch.zeros((num_steps, num_envs, global_obs_dim), device=device)  # 全局观测
        self.actions = torch.zeros((num_steps, num_envs, num_agents, action_dim), device=device)  # 动作
        self.log_probs = torch.zeros((num_steps, num_envs, num_agents), device=device)  # 动作对数概率
        self.rewards = torch.zeros((num_steps, num_envs), device=device)  # 共享奖励
        self.dones = torch.zeros((num_steps, num_envs), device=device)  # 终止标志
        self.values = torch.zeros((num_steps, num_envs), device=device)  # 价值估计
        self.advantages = torch.zeros((num_steps, num_envs), device=device)  # GAE优势
        self.returns = torch.zeros((num_steps, num_envs), device=device)  # 回报
    
    def store(self, obs, global_obs, actions, log_probs, rewards, dones, values):
        """存储一步转移数据"""
        self.obs[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr = (self.ptr + 1) % self.num_steps
    
    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """
        计算GAE (Generalized Advantage Estimation)
        GAE平衡了偏差和方差，是PPO的关键组件
        gamma: 折扣因子
        lam: GAE平滑参数
        """
        gae = torch.zeros((self.num_envs,), device=self.device)
        
        # 从后向前计算GAE
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            
            # TD误差: δ = r + γV(s') - V(s)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            # GAE递推: A_t = δ_t + γλA_{t+1}
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]  # 回报 = 优势 + 价值
        
        # 标准化优势，稳定训练
        adv_flat = self.advantages.view(-1)
        self.advantages = (self.advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    
    def get_batches(self, batch_size):
        """生成随机小批量用于训练"""
        total_samples = self.num_steps * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]
            
            # 将一维索引转换为二维索引
            step_idx = batch_indices // self.num_envs
            env_idx = batch_indices % self.num_envs
            
            yield {
                'obs': self.obs[step_idx, env_idx],
                'global_obs': self.global_obs[step_idx, env_idx],
                'actions': self.actions[step_idx, env_idx],
                'log_probs': self.log_probs[step_idx, env_idx],
                'advantages': self.advantages[step_idx, env_idx],
                'returns': self.returns[step_idx, env_idx],
            }
    
    def clear(self):
        """重置缓冲区指针"""
        self.ptr = 0


class MAPPO:
    """MAPPO算法主类，整合Actor、Critic和训练逻辑"""
    
    def __init__(
        self,
        num_agents,
        obs_dim,
        global_obs_dim,
        action_dim,
        device,
        lr_actor=3e-4,       # Actor学习率
        lr_critic=5e-4,      # Critic学习率（通常略高）
        gamma=0.99,          # 折扣因子
        lam=0.95,            # GAE参数
        clip_param=0.2,      # PPO裁剪范围
        entropy_coef=0.02,   # 熵正则化系数（鼓励探索）
        value_loss_coef=0.5, # 价值损失权重
        max_grad_norm=1.0,   # 梯度裁剪阈值
        num_epochs=5,        # 每次更新的epoch数
        batch_size=512,      # 小批量大小
        share_actor=True,    # 是否共享Actor参数（同质智能体）
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # 保存超参数
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.share_actor = share_actor
        
        # 网络结构配置
        actor_hidden = [256, 256, 128]
        critic_hidden = [512, 256, 128]
        
        # 创建Actor网络
        if share_actor:
            # 参数共享：所有智能体使用同一个Actor（适用于同质智能体）
            self.actor = MAPPOActor(obs_dim, action_dim, hidden_dims=actor_hidden, init_std=0.5).to(device)
            self.actors = [self.actor] * num_agents
        else:
            # 独立Actor：每个智能体有自己的策略（适用于异质智能体）
            self.actors = [MAPPOActor(obs_dim, action_dim, hidden_dims=actor_hidden, init_std=0.5).to(device) 
                          for _ in range(num_agents)]
        
        # 创建集中式Critic（所有智能体共享）
        self.critic = MAPPOCritic(global_obs_dim, hidden_dims=critic_hidden).to(device)
        
        # 配置优化器
        if share_actor:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        else:
            actor_params = []
            for actor in self.actors:
                actor_params.extend(actor.parameters())
            self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def get_actions(self, obs, deterministic=False):
        """
        获取所有智能体的动作
        obs: [num_envs, num_agents, obs_dim]
        返回: actions, log_probs
        """
        num_envs = obs.shape[0]
        actions = torch.zeros((num_envs, self.num_agents, self.action_dim), device=self.device)
        log_probs = torch.zeros((num_envs, self.num_agents), device=self.device)
        
        # 每个智能体独立决策
        for i, actor in enumerate(self.actors):
            agent_obs = obs[:, i, :]
            act, lp = actor.get_action(agent_obs, deterministic=deterministic)
            actions[:, i, :] = act
            log_probs[:, i] = lp
        
        return actions, log_probs
    
    def get_value(self, global_obs):
        """使用Critic评估全局状态价值"""
        return self.critic(global_obs)
    
    def update(self, buffer):
        """
        PPO更新步骤
        1. 计算GAE优势
        2. 多轮epoch更新Actor和Critic
        """
        # 计算GAE
        buffer.compute_gae(
            self.get_value(buffer.global_obs[-1]).detach(),
            self.gamma,
            self.lam
        )
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # 多轮epoch更新
        for _ in range(self.num_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs = batch['obs']
                global_obs = batch['global_obs']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # ========== Actor更新 ==========
                new_log_probs = torch.zeros_like(old_log_probs)
                entropy = torch.zeros_like(old_log_probs)
                
                # 计算新策略下的对数概率
                for i, actor in enumerate(self.actors):
                    agent_obs = obs[:, i, :]
                    agent_actions = actions[:, i, :]
                    lp, ent = actor.evaluate_actions(agent_obs, agent_actions)
                    new_log_probs[:, i] = lp
                    entropy[:, i] = ent
                
                # 计算重要性采样比率（联合策略）
                ratio = torch.exp(new_log_probs.sum(dim=-1) - old_log_probs.sum(dim=-1))
                
                # PPO-Clip目标函数
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 熵正则化（鼓励探索）
                entropy_loss = -entropy.mean()
                
                # Actor总损失
                total_actor_loss_batch = actor_loss + self.entropy_coef * entropy_loss
                
                # 反向传播和梯度更新
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                if self.share_actor:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                else:
                    for actor in self.actors:
                        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # ========== Critic更新 ==========
                values = self.critic(global_obs)
                # MSE损失
                critic_loss = self.value_loss_coef * ((values - returns) ** 2).mean()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # 记录统计信息
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        buffer.clear()
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def save(self, path):
        """保存模型参数"""
        state = {'critic': self.critic.state_dict()}
        if self.share_actor:
            state['actor'] = self.actor.state_dict()
        else:
            for i, actor in enumerate(self.actors):
                state[f'actor_{i}'] = actor.state_dict()
        torch.save(state, path)
    
    def load(self, path):
        """加载模型参数"""
        state = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(state['critic'])
        if self.share_actor:
            self.actor.load_state_dict(state['actor'])
        else:
            for i, actor in enumerate(self.actors):
                actor.load_state_dict(state[f'actor_{i}'])
