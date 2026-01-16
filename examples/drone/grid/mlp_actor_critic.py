"""
纯MLP架构的Actor-Critic网络（用于障碍物避障）

该模块实现了一个简单的多层感知机（MLP）网络，用于单无人机避障任务。
与CNN+MLP混合架构不同，这个网络将所有观测（自身状态 + 障碍物网格展平）
统一用MLP处理，架构更简单，适合：
1. 障碍物信息已经被编码为距离衰减值（不需要空间卷积）
2. 需要快速训练和推理
3. 观测维度不是很高（68维）

网络架构图：
┌─────────────────────────────────────────────────────────────────┐
│                      观测输入 (68维)                             │
│              自身状态(19) + 障碍物网格展平(49)                     │
│                            │                                     │
│                            ▼                                     │
│                  ┌───────────────────┐                           │
│                  │   共享特征提取     │                           │
│                  │   MLP Backbone    │                           │
│                  │  Linear -> ELU    │                           │
│                  │  Linear -> ELU    │                           │
│                  │  Linear -> ELU    │                           │
│                  └─────────┬─────────┘                           │
│                           │ 256维                                │
│              ┌────────────┴────────────┐                         │
│              ▼                         ▼                         │
│      ┌───────────────┐         ┌───────────────┐                │
│      │  Actor头      │         │  Critic头     │                │
│      │ Linear->ELU   │         │ Linear->ELU   │                │
│      │   Linear      │         │   Linear      │                │
│      └───────┬───────┘         └───────┬───────┘                │
│              ▼                         ▼                         │
│        动作均值 (4维)              价值估计 (1维)                 │
└─────────────────────────────────────────────────────────────────┘

与rsl_rl库的集成：
- 继承自nn.Module，实现rsl_rl要求的接口
- 包含act(), act_inference(), evaluate()等方法
- 兼容OnPolicyRunner的训练流程

依赖：
- PyTorch
- rsl_rl (rsl-rl-lib==2.2.4)
"""
import torch
import torch.nn as nn


class MLPActorCritic(nn.Module):
    """
    纯MLP架构的Actor-Critic网络

    该网络使用简单的多层感知机处理所有观测信息，相比CNN+MLP混合架构：
    - 优势：更简单、更快、参数更少、易于训练
    - 适用场景：障碍物信息已编码为特征向量（距离衰减值）

    Attributes:
        num_obs (int): 总观测维度（自身状态 + 障碍物网格展平）
        num_actions (int): 动作维度（4个电机）
        is_recurrent (bool): 是否为循环网络（False）
        backbone (nn.Sequential): 共享特征提取器
        actor (nn.Sequential): Actor策略头
        critic (nn.Sequential): Critic价值头
        std (nn.Parameter): 动作标准差参数
    """

    def __init__(
        self,
        num_obs,                     # 总观测维度（68维：19+49）
        num_actions,                 # 动作维度（4）
        backbone_hidden_dims=[256, 256, 256],  # 共享backbone隐藏层维度
        actor_hidden_dims=[128],     # Actor头隐藏层维度
        critic_hidden_dims=[128],    # Critic头隐藏层维度
        activation='elu',            # 激活函数类型
        init_noise_std=1.0,          # 初始动作噪声标准差
    ):
        """
        初始化纯MLP网络

        Args:
            num_obs (int): 总观测维度
                - 自身状态(19) + 障碍物网格展平(49) = 68
            num_actions (int): 动作空间维度（4个电机转速）
            backbone_hidden_dims (list): 共享特征提取器的隐藏层维度
                - 默认[256, 256, 256]，三层全连接
            actor_hidden_dims (list): Actor头隐藏层维度
            critic_hidden_dims (list): Critic头隐藏层维度
            activation (str): 激活函数类型，'elu'或'relu'
            init_noise_std (float): 动作标准差的初始值
        """
        super().__init__()

        # 保存基本参数
        self.num_obs = num_obs
        self.num_actions = num_actions

        # ==================== rsl_rl 接口所需属性 ====================
        self.is_recurrent = False  # 非循环网络
        self.num_actor_obs = num_obs
        self.num_critic_obs = num_obs

        # 动作分布相关属性
        self.action_mean = None
        self.action_std = None
        self.entropy = None

        # ==================== 激活函数选择 ====================
        activation_fn = nn.ELU() if activation == 'elu' else nn.ReLU()

        # ==================== 共享特征提取器（Backbone） ====================
        """
        Backbone设计：
        - 输入：68维观测向量（自身状态 + 障碍物网格展平）
        - 三层全连接：68 -> 256 -> 256 -> 256
        - 输出：256维特征向量

        这个共享的backbone提取特征，Actor和Critic共享这些特征
        """
        backbone_layers = []
        backbone_input_dim = num_obs
        for hidden_dim in backbone_hidden_dims:
            backbone_layers.append(nn.Linear(backbone_input_dim, hidden_dim))
            backbone_layers.append(activation_fn)
            backbone_input_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)
        backbone_output_dim = backbone_hidden_dims[-1]  # 256

        # ==================== Actor头（策略网络） ====================
        """
        Actor网络设计：
        - 输入：256维backbone特征
        - 一层全连接：256 -> 128 -> 4
        - 输出：4维动作均值（4个电机转速）
        """
        actor_layers = []
        actor_input_dim = backbone_output_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(actor_input_dim, hidden_dim))
            actor_layers.append(activation_fn)
            actor_input_dim = hidden_dim
        # 最后一层输出动作维度（无激活函数）
        actor_layers.append(nn.Linear(actor_input_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # ==================== Critic头（价值网络） ====================
        """
        Critic网络设计：
        - 输入：256维backbone特征
        - 一层全连接：256 -> 128 -> 1
        - 输出：1维状态价值估计V(s)
        """
        critic_layers = []
        critic_input_dim = backbone_output_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_input_dim, hidden_dim))
            critic_layers.append(activation_fn)
            critic_input_dim = hidden_dim
        # 最后一层输出1维价值
        critic_layers.append(nn.Linear(critic_input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ==================== 动作标准差 ====================
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # 初始化网络权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化网络权重

        使用正交初始化（Orthogonal Initialization）
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        """
        前向传播：提取特征

        Args:
            observations (torch.Tensor): 观测张量
                形状: (batch_size, num_obs) = (batch, 68)

        Returns:
            torch.Tensor: Backbone特征向量
                形状: (batch_size, 256)
        """
        # 通过共享backbone提取特征
        features = self.backbone(observations)
        return features

    def act(self, observations, masks=None, hidden_states=None):
        """
        采样动作（训练时使用）

        Args:
            observations (torch.Tensor): 观测张量，形状 (batch, 68)
            masks: 掩码（非循环网络不使用）
            hidden_states: 隐藏状态（非循环网络不使用）

        Returns:
            torch.Tensor: 采样的动作，形状 (batch, 4)

        副作用：
        - 更新 self.action_mean, self.action_std, self.distribution, self.entropy
        """
        # 提取特征
        features = self.forward(observations)

        # 通过Actor网络获取动作均值
        action_mean = self.actor(features)

        # 保存属性（rsl_rl需要访问）
        self.action_mean = action_mean
        self.action_std = self.std

        # 创建高斯分布并采样
        self.distribution = torch.distributions.Normal(action_mean, self.std)
        actions = self.distribution.sample()

        # 计算熵
        self.entropy = self.distribution.entropy().sum(dim=-1)

        return actions

    def act_inference(self, observations):
        """
        确定性动作（评估/部署时使用）

        Args:
            observations (torch.Tensor): 观测张量，形状 (batch, 68)

        Returns:
            torch.Tensor: 确定性动作（均值），形状 (batch, 4)
        """
        features = self.forward(observations)
        action_mean = self.actor(features)
        return action_mean

    def evaluate(self, observations, actions=None, masks=None, hidden_states=None):
        """
        评估观测和动作（PPO更新时使用）

        Args:
            observations (torch.Tensor): 观测张量，形状 (batch, 68)
            actions (torch.Tensor, optional): 动作张量，形状 (batch, 4)
            masks: 掩码（非循环网络不使用）
            hidden_states: 隐藏状态（非循环网络不使用）

        Returns:
            如果 actions 为 None:
                torch.Tensor: 状态价值，形状 (batch, 1)
            如果 actions 不为 None:
                tuple: (log_prob, entropy, value)
        """
        # 提取特征
        features = self.forward(observations)

        # Critic评估：计算状态价值
        value = self.critic(features)  # 保持 [batch, 1] 的形状

        # 如果没有提供actions，只返回value
        if actions is None:
            return value

        # 如果提供了actions，计算完整的PPO更新所需信息
        action_mean = self.actor(features)
        self.distribution = torch.distributions.Normal(action_mean, self.std)

        # 计算给定动作的对数概率
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)

        # 计算分布熵
        entropy = self.distribution.entropy().sum(dim=-1)

        return actions_log_prob, entropy, value

    def get_actions_log_prob(self, actions):
        """
        获取动作的对数概率

        Args:
            actions (torch.Tensor): 动作张量，形状 (batch, 4)

        Returns:
            torch.Tensor: 对数概率，形状 (batch,)
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        """
        重置网络状态（非循环网络为空操作）

        Args:
            dones: 完成标志
        """
        pass
