"""
自定义CNN+MLP混合架构的Actor-Critic网络
用于单无人机避障：CNN处理3D障碍物网格，MLP处理自身状态
"""
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic


class CNNMLPActorCritic(nn.Module):
    """
    混合架构的Actor-Critic网络
    - CNN: 处理3x3x3障碍物感知立方体
    - MLP: 处理自身状态(位置、速度、目标距离、目标方位角等)
    - 特征融合后输出动作和价值估计
    """

    def __init__(
        self,
        num_state_obs,       # 自身状态维度
        num_actions,         # 动作维度
        grid_size=3,         # 障碍物网格大小
        cnn_channels=[16, 32, 64],  # CNN通道数
        mlp_hidden_dims=[128, 128],  # MLP隐藏层维度
        actor_hidden_dims=[256, 128],  # Actor头隐藏层
        critic_hidden_dims=[256, 128],  # Critic头隐藏层
        activation='elu',
        init_noise_std=1.0,
    ):
        super().__init__()

        self.num_state_obs = num_state_obs
        self.num_actions = num_actions
        self.grid_size = grid_size
        self.grid_dim = grid_size ** 3  # 27

        # rsl_rl 需要的属性
        self.is_recurrent = False
        self.num_actor_obs = num_state_obs + self.grid_dim
        self.num_critic_obs = num_state_obs + self.grid_dim

        # 初始化动作相关属性（rsl_rl 会访问）
        self.action_mean = None
        self.action_std = None
        self.entropy = None

        # 激活函数
        activation_fn = nn.ELU() if activation == 'elu' else nn.ReLU()

        # ==================== CNN部分：处理3x3x3障碍物网格 ====================
        # 输入: (batch, 1, 3, 3, 3)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, cnn_channels[0], kernel_size=3, padding=1),
            activation_fn,
            nn.Conv3d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            activation_fn,
            nn.Conv3d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            activation_fn,
            nn.Flatten(),  # 输出: (batch, 64*3*3*3) = (batch, 1728)
        )
        cnn_output_dim = cnn_channels[-1] * 3 * 3 * 3

        # ==================== MLP部分：处理自身状态 ====================
        mlp_layers = []
        mlp_input_dim = num_state_obs
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_layers.append(activation_fn)
            mlp_input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_output_dim = mlp_hidden_dims[-1]

        # ==================== 特征融合 ====================
        fusion_dim = cnn_output_dim + mlp_output_dim

        # ==================== Actor头 ====================
        actor_layers = []
        actor_input_dim = fusion_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(actor_input_dim, hidden_dim))
            actor_layers.append(activation_fn)
            actor_input_dim = hidden_dim
        actor_layers.append(nn.Linear(actor_input_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # ==================== Critic头 ====================
        critic_layers = []
        critic_input_dim = fusion_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_input_dim, hidden_dim))
            critic_layers.append(activation_fn)
            critic_input_dim = hidden_dim
        critic_layers.append(nn.Linear(critic_input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ==================== 动作标准差 ====================
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # 初始化网络权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        """
        前向传播
        observations: 展平的tensor (batch, num_state_obs + grid_dim)
        """
        # 拆分观测
        state = observations[:, :self.num_state_obs]  # (batch, 19)
        grid_flat = observations[:, self.num_state_obs:]  # (batch, 27)

        # 将展平的网格重塑为3D
        obstacle_grid = grid_flat.reshape(-1, self.grid_size, self.grid_size, self.grid_size)  # (batch, 3, 3, 3)

        # CNN处理障碍物网格
        obstacle_grid = obstacle_grid.unsqueeze(1)  # (batch, 1, 3, 3, 3)
        cnn_features = self.cnn(obstacle_grid)  # (batch, 1728)

        # MLP处理自身状态
        mlp_features = self.mlp(state)  # (batch, 128)

        # 特征融合
        fused_features = torch.cat([cnn_features, mlp_features], dim=-1)

        return fused_features

    def act(self, observations, masks=None, hidden_states=None):
        """
        采样动作（训练时使用）
        返回：actions（张量）

        Args:
            observations: 观测值
            masks: 掩码（rsl_rl 接口需要，但我们的非循环网络不使用）
            hidden_states: 隐藏状态（rsl_rl 接口需要，但我们的非循环网络不使用）
        """
        fused_features = self.forward(observations)
        action_mean = self.actor(fused_features)

        # 保存 action_mean 和 action_std（rsl_rl 需要访问）
        self.action_mean = action_mean
        self.action_std = self.std

        # 创建正态分布并保存（用于后续计算 log_prob）
        self.distribution = torch.distributions.Normal(action_mean, self.std)
        actions = self.distribution.sample()

        # 计算并保存 entropy（rsl_rl 需要访问）
        self.entropy = self.distribution.entropy().sum(dim=-1)

        return actions  # 只返回 actions

    def act_inference(self, observations):
        """
        确定性动作（评估时使用）
        """
        fused_features = self.forward(observations)
        action_mean = self.actor(fused_features)
        return action_mean

    def evaluate(self, observations, actions=None, masks=None, hidden_states=None):
        """
        评估观测和动作
        - 如果只传 observations：返回 value（形状 [num_envs, 1]）
        - 如果传 observations + actions：返回 (log_probs, entropy, value)（三个张量）

        注意：rsl_rl 的存储期望 value 的形状是 [num_envs, 1]，不要 squeeze！

        Args:
            observations: 观测值
            actions: 动作（可选）
            masks: 掩码（rsl_rl 接口需要，但我们的非循环网络不使用）
            hidden_states: 隐藏状态（rsl_rl 接口需要，但我们的非循环网络不使用）
        """
        fused_features = self.forward(observations)

        # Critic评估（总是需要）
        # 保持 [batch, 1] 的形状，不要 squeeze！
        value = self.critic(fused_features)

        # 如果没有提供 actions，只返回 value
        if actions is None:
            return value

        # Actor评估（PPO更新时使用）
        action_mean = self.actor(fused_features)
        self.distribution = torch.distributions.Normal(action_mean, self.std)
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        entropy = self.distribution.entropy().sum(dim=-1)

        # value 保持 [batch, 1] 形状
        return actions_log_prob, entropy, value

    def get_actions_log_prob(self, actions):
        """获取动作的对数概率"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        """
        重置网络状态（用于循环网络）
        我们的网络不是循环的，所以这是一个空操作
        """
        pass
