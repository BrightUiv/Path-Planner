"""
自定义CNN+MLP混合架构的Actor-Critic网络

该模块实现了一个专门用于单无人机避障任务的神经网络架构。
网络采用双分支设计：
1. CNN分支：处理3D障碍物感知网格（空间特征提取）
2. MLP分支：处理无人机自身状态（位置、速度、姿态等）

障碍物感知采用线性衰减方案：
- 公式：v = max(0, 1 - d / d_max)
- d=0（接触）时 v=1，d=d_max时 v=0，d>d_max时 v=0
- 值越大表示障碍物越近，策略应更积极地避障

网络架构图（7x7x3网格版本）：
┌─────────────────────────────────────────────────────────────────┐
│                      观测输入 (166维)                            │
│              ┌─────────────────┴─────────────────┐              │
│              ▼                                   ▼              │
│      自身状态 (19维)              障碍物网格 (147维, 线性衰减值)   │
│              │                                   │              │
│              ▼                                   ▼              │
│      ┌───────────────┐                  ┌───────────────┐       │
│      │   MLP分支     │                  │   CNN分支     │       │
│      │ Linear->ELU   │                  │ Conv3D->ELU   │       │
│      │ Linear->ELU   │                  │ Conv3D->ELU   │       │
│      └───────┬───────┘                  │ AdaptivePool  │       │
│              │                          │   Flatten     │       │
│              │                          └───────┬───────┘       │
│              │           128维                  │ 256维         │
│              └─────────────┬────────────────────┘              │
│                           ▼                                     │
│                    特征融合 (384维)                              │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                       │
│      ┌───────────────┐         ┌───────────────┐               │
│      │  Actor头      │         │  Critic头     │               │
│      │ Linear->ELU   │         │ Linear->ELU   │               │
│      │ Linear->ELU   │         │ Linear->ELU   │               │
│      │   Linear      │         │   Linear      │               │
│      └───────┬───────┘         └───────┬───────┘               │
│              ▼                         ▼                       │
│        动作均值 (4维)              价值估计 (1维)                │
└─────────────────────────────────────────────────────────────────┘

与rsl_rl库的集成：
- 该网络继承自nn.Module，但实现了rsl_rl要求的接口
- 包含act(), act_inference(), evaluate()等方法
- 兼容OnPolicyRunner的训练流程

依赖：
- PyTorch
- rsl_rl (rsl-rl-lib==2.2.4)
"""
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic


class CNNMLPActorCritic(nn.Module):
    """
    混合架构的Actor-Critic网络

    该网络专门为无人机避障任务设计，结合了：
    - 3D卷积神经网络(CNN)：提取障碍物空间分布的特征
    - 多层感知机(MLP)：处理无人机的运动学状态

    这种混合架构的优势：
    1. CNN能够捕捉障碍物的空间结构信息
    2. MLP能够高效处理连续的状态向量
    3. 特征融合后可以学习状态与障碍物之间的关系

    Attributes:
        num_state_obs (int): 自身状态观测维度（19）
        num_actions (int): 动作维度（4个电机）
        grid_size (int): 障碍物网格大小（3）
        grid_dim (int): 网格展平维度（27）
        is_recurrent (bool): 是否为循环网络（False）
        cnn (nn.Sequential): CNN特征提取器
        mlp (nn.Sequential): MLP特征提取器
        actor (nn.Sequential): Actor策略头
        critic (nn.Sequential): Critic价值头
        std (nn.Parameter): 动作标准差参数
    """

    def __init__(
        self,
        num_state_obs,       # 自身状态维度 (19)
        num_actions,         # 动作维度 (4)
        grid_size=(7, 7, 3),  # 障碍物网格形状 (X, Y, Z)
        cnn_channels=[32, 64],  # CNN各层通道数
        mlp_hidden_dims=[128, 128],  # MLP隐藏层维度
        actor_hidden_dims=[256, 128],  # Actor头隐藏层维度
        critic_hidden_dims=[256, 128],  # Critic头隐藏层维度
        activation='elu',    # 激活函数类型
        init_noise_std=1.0,  # 初始动作噪声标准差
    ):
        """
        初始化CNN+MLP混合网络

        Args:
            num_state_obs (int): 自身状态观测的维度
                - 位置(3) + 速度(3) + 距离(1) + 方位角(1) + 四元数(4) + 角速度(3) + 上一动作(4) = 19
            num_actions (int): 动作空间维度（4个电机转速）
            grid_size (tuple): 3D障碍物网格的形状 (X, Y, Z)
                - 默认(7, 7, 3)，水平方向更宽，垂直方向较窄
            cnn_channels (list): CNN每层的输出通道数
                - 默认[32, 64]，两层卷积
            mlp_hidden_dims (list): 状态MLP每层的隐藏单元数
                - 默认[128, 128]，两层全连接
            actor_hidden_dims (list): Actor头每层的隐藏单元数
            critic_hidden_dims (list): Critic头每层的隐藏单元数
            activation (str): 激活函数类型，'elu'或'relu'
            init_noise_std (float): 动作标准差的初始值
                - 较大的值（如1.0）鼓励早期探索
                - 训练过程中会自动调整
        """
        super().__init__()

        # 保存基本参数
        self.num_state_obs = num_state_obs
        self.num_actions = num_actions
        # 支持元组或整数形式的 grid_size
        if isinstance(grid_size, int):
            self.grid_shape = (grid_size, grid_size, grid_size)
        else:
            self.grid_shape = tuple(grid_size)
        self.grid_size_x, self.grid_size_y, self.grid_size_z = self.grid_shape
        self.grid_dim = self.grid_size_x * self.grid_size_y * self.grid_size_z  # 7*7*3 = 147

        # ==================== rsl_rl 接口所需属性 ====================
        # rsl_rl库在训练时会访问这些属性
        self.is_recurrent = False  # 非循环网络（没有LSTM/GRU）
        self.num_actor_obs = num_state_obs + self.grid_dim  # Actor观测维度: 19+147=166
        self.num_critic_obs = num_state_obs + self.grid_dim  # Critic观测维度: 19+147=166

        # 动作分布相关属性（rsl_rl会访问）
        self.action_mean = None  # 动作均值
        self.action_std = None   # 动作标准差
        self.entropy = None      # 动作熵（用于鼓励探索）

        # ==================== 激活函数选择 ====================
        # ELU(Exponential Linear Unit)比ReLU更平滑，有助于梯度流动
        activation_fn = nn.ELU() if activation == 'elu' else nn.ReLU()

        # ==================== CNN部分：处理7x7x3障碍物网格 ====================
        """
        3D卷积网络设计（适应非立方体输入）：
        - 输入: (batch, 1, 7, 7, 3) - 单通道的3D网格
        - 两层卷积逐步增加通道数：1 -> 32 -> 64
        - 使用自适应池化统一输出大小
        - 最后展平为向量：64 * 2 * 2 * 2 = 512维
        """
        self.cnn = nn.Sequential(
            # 第一层卷积：1 -> 32通道，kernel_size=3, padding=1保持大小
            nn.Conv3d(1, cnn_channels[0], kernel_size=3, padding=1),
            activation_fn,
            # 第二层卷积：32 -> 64通道
            nn.Conv3d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            activation_fn,
            # 自适应平均池化：将任意大小的输入统一到 (2, 2, 2)
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            # 展平：(batch, 64, 2, 2, 2) -> (batch, 512)
            nn.Flatten(),
        )
        cnn_output_dim = cnn_channels[-1] * 2 * 2 * 2  # 64 * 8 = 512

        # ==================== MLP部分：处理自身状态 ====================
        """
        状态MLP设计：
        - 输入：19维状态向量
        - 两层全连接：19 -> 128 -> 128
        - 输出：128维特征向量
        """
        mlp_layers = []
        mlp_input_dim = num_state_obs  # 19
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_layers.append(activation_fn)
            mlp_input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_output_dim = mlp_hidden_dims[-1]  # 128

        # ==================== 特征融合 ====================
        # 将CNN和MLP的输出拼接：1728 + 128 = 1856维
        fusion_dim = cnn_output_dim + mlp_output_dim

        # ==================== Actor头（策略网络） ====================
        """
        Actor网络设计：
        - 输入：1856维融合特征
        - 两层全连接：1856 -> 256 -> 128 -> 4
        - 输出：4维动作均值（4个电机转速）

        Actor输出的是动作的均值，实际动作会加上高斯噪声：
        action = mean + std * N(0, 1)
        """
        actor_layers = []
        actor_input_dim = fusion_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(actor_input_dim, hidden_dim))
            actor_layers.append(activation_fn)
            actor_input_dim = hidden_dim
        # 最后一层输出动作维度（无激活函数，输出可以是任意实数）
        actor_layers.append(nn.Linear(actor_input_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # ==================== Critic头（价值网络） ====================
        """
        Critic网络设计：
        - 输入：1856维融合特征
        - 两层全连接：1856 -> 256 -> 128 -> 1
        - 输出：1维状态价值估计V(s)

        Critic估计当前状态的期望累积奖励，用于：
        1. 计算优势函数A(s,a) = Q(s,a) - V(s)
        2. 减少策略梯度的方差
        """
        critic_layers = []
        critic_input_dim = fusion_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_input_dim, hidden_dim))
            critic_layers.append(activation_fn)
            critic_input_dim = hidden_dim
        # 最后一层输出1维价值
        critic_layers.append(nn.Linear(critic_input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ==================== 动作标准差 ====================
        """
        动作标准差设计：
        - 使用可学习参数（nn.Parameter）
        - 所有动作维度共享同一个标准差（也可以设计为独立的）
        - 初始值较大（1.0）鼓励早期探索
        - 训练过程中会自动减小，使策略更确定
        """
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None  # 当前的动作分布（用于计算log_prob和entropy）

        # 初始化网络权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化网络权重

        使用正交初始化（Orthogonal Initialization）：
        - 对于ReLU/ELU网络，正交初始化有助于保持梯度的稳定传播
        - gain=1.0 是标准的缩放因子
        - bias初始化为0
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        """
        前向传播：提取融合特征

        该方法只做特征提取，不输出动作或价值。
        被act()和evaluate()内部调用。

        Args:
            observations (torch.Tensor): 展平的观测张量
                形状: (batch_size, num_state_obs + grid_dim) = (batch, 166)

        Returns:
            torch.Tensor: 融合后的特征向量
                形状: (batch_size, 640) = 512(CNN) + 128(MLP)

        处理流程：
        1. 拆分观测为状态和网格
        2. CNN处理网格 -> 512维特征
        3. MLP处理状态 -> 128维特征
        4. 拼接特征 -> 640维
        """
        # 拆分观测：前19维是状态，后147维是网格
        state = observations[:, :self.num_state_obs]  # (batch, 19)
        grid_flat = observations[:, self.num_state_obs:]  # (batch, 147)

        # 将展平的网格重塑为3D形状
        # (batch, 147) -> (batch, 7, 7, 3)
        obstacle_grid = grid_flat.reshape(-1, self.grid_size_x, self.grid_size_y, self.grid_size_z)

        # CNN处理障碍物网格
        # (batch, 7, 7, 3) -> (batch, 1, 7, 7, 3) 添加通道维度
        obstacle_grid = obstacle_grid.unsqueeze(1)
        # (batch, 1, 7, 7, 3) -> (batch, 512)
        cnn_features = self.cnn(obstacle_grid)

        # MLP处理自身状态
        # (batch, 19) -> (batch, 128)
        mlp_features = self.mlp(state)

        # 特征融合：拼接CNN和MLP的输出
        # (batch, 512) + (batch, 128) -> (batch, 640)
        fused_features = torch.cat([cnn_features, mlp_features], dim=-1)

        return fused_features

    def act(self, observations, masks=None, hidden_states=None):
        """
        采样动作（训练时使用）

        在训练过程中，需要从策略分布中采样动作，并记录相关信息
        用于后续的PPO更新。

        Args:
            observations (torch.Tensor): 观测张量，形状 (batch, 46)
            masks: 掩码（rsl_rl接口需要，非循环网络不使用）
            hidden_states: 隐藏状态（rsl_rl接口需要，非循环网络不使用）

        Returns:
            torch.Tensor: 采样的动作，形状 (batch, 4)

        副作用：
        - 更新 self.action_mean: 动作均值
        - 更新 self.action_std: 动作标准差
        - 更新 self.distribution: 当前的动作分布
        - 更新 self.entropy: 动作熵
        """
        # 提取融合特征
        fused_features = self.forward(observations)

        # 通过Actor网络获取动作均值
        action_mean = self.actor(fused_features)

        # 保存 action_mean 和 action_std（rsl_rl 需要访问这些属性）
        self.action_mean = action_mean
        self.action_std = self.std

        # 创建高斯分布：N(action_mean, std)
        self.distribution = torch.distributions.Normal(action_mean, self.std)

        # 从分布中采样动作
        # 使用重参数化技巧：action = mean + std * epsilon, epsilon ~ N(0, 1)
        actions = self.distribution.sample()

        # 计算并保存熵（用于PPO的熵正则化）
        # 熵越大表示策略越不确定，可以鼓励探索
        self.entropy = self.distribution.entropy().sum(dim=-1)

        return actions  # 只返回动作，rsl_rl会通过属性访问其他信息

    def act_inference(self, observations):
        """
        确定性动作（评估/部署时使用）

        在评估或实际部署时，不需要探索，直接使用策略的均值作为动作。
        这样可以得到更稳定、更优的行为。

        Args:
            observations (torch.Tensor): 观测张量，形状 (batch, 46)

        Returns:
            torch.Tensor: 确定性动作（均值），形状 (batch, 4)
        """
        fused_features = self.forward(observations)
        action_mean = self.actor(fused_features)
        return action_mean  # 直接返回均值，不添加噪声

    def evaluate(self, observations, actions=None, masks=None, hidden_states=None):
        """
        评估观测和动作（PPO更新时使用）

        这个方法有两种使用模式：
        1. 只传observations：返回状态价值V(s)
        2. 传observations和actions：返回(log_prob, entropy, value)

        Args:
            observations (torch.Tensor): 观测张量，形状 (batch, 46)
            actions (torch.Tensor, optional): 动作张量，形状 (batch, 4)
            masks: 掩码（rsl_rl接口需要，非循环网络不使用）
            hidden_states: 隐藏状态（rsl_rl接口需要，非循环网络不使用）

        Returns:
            如果 actions 为 None:
                torch.Tensor: 状态价值，形状 (batch, 1)
            如果 actions 不为 None:
                tuple: (log_prob, entropy, value)
                    - log_prob: 动作的对数概率，形状 (batch,)
                    - entropy: 分布熵，形状 (batch,)
                    - value: 状态价值，形状 (batch, 1)

        注意：
        - rsl_rl的存储期望value的形状是[batch, 1]，不要squeeze！
        - log_prob需要对所有动作维度求和
        """
        # 提取融合特征
        fused_features = self.forward(observations)

        # Critic评估：计算状态价值
        # 保持 [batch, 1] 的形状，不要 squeeze！
        value = self.critic(fused_features)

        # 如果没有提供 actions，只返回 value
        # 这用于收集轨迹时估计状态价值
        if actions is None:
            return value

        # 如果提供了 actions，需要计算完整的PPO更新所需信息
        # Actor评估：重新计算动作分布
        action_mean = self.actor(fused_features)
        self.distribution = torch.distributions.Normal(action_mean, self.std)

        # 计算给定动作的对数概率
        # log_prob(a) = sum_i log p(a_i | s)
        actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)

        # 计算分布熵
        # H = -E[log p(a|s)] = sum_i H_i
        entropy = self.distribution.entropy().sum(dim=-1)

        # value 保持 [batch, 1] 形状
        return actions_log_prob, entropy, value

    def get_actions_log_prob(self, actions):
        """
        获取动作的对数概率

        使用当前保存的分布计算给定动作的log概率。
        必须在调用act()之后使用，因为需要self.distribution。

        Args:
            actions (torch.Tensor): 动作张量，形状 (batch, 4)

        Returns:
            torch.Tensor: 对数概率，形状 (batch,)
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        """
        重置网络状态

        对于循环网络（如LSTM），需要在episode结束时重置隐藏状态。
        我们的网络不是循环的，所以这是一个空操作。

        Args:
            dones: 完成标志（用于部分重置，循环网络使用）
        """
        pass  # 非循环网络，无需重置
