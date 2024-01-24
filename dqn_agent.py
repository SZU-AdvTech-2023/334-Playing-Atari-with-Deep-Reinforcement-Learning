import numpy as np
import random
from collections import namedtuple, deque

from q_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)  # behavior network
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)  # target network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR) #优化器

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    #添加信息
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # 添加进来的也是四个不同的帧，有可能会和replaybuffer有重复帧,
        #其实也可以一步学一次，反正都是抽样
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY # 每四步学一次 四个不同帧,
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) #变成四维数据 (1*4*84*84)
        self.qnetwork_local.eval() # 为什么要加这段代码
        # eval() 更多关注于改变模型的行为（特别是特定层的行为），而 no_grad() 更多关注于计算效率，阻止梯度计算和存储。
        # 共同使用：在评估模型时，通常会同时使用这两者。先调用 eval() 确保模型在评估模式，然后在 no_grad() 的上下文中进行前向传播，以减少计算和内存开销。
        # 不同的应用场景：即使在模型训练阶段，有时也会使用 no_grad()，比如在计算某些不需要梯度的指标时。但在这种情况下，模型仍然保持在训练模式（train()）
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # 返回的是索引
        if random.random() > eps:
            #使用 .data 可以避免 PyTorch 跟踪张量上的操作，这在某些情况下有助于防止意外修改梯度。
            return np.argmax(action_values.cpu().data.numpy()) #这里为什么需要这么复杂 不能直接argmax
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #todo below is double-dqn
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)  # (b,1)
        # 使用本地网络选择下一个状态的动作，但不进行梯度计算

        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_targets_next_actions) # (b,4) 对的是Q_targets_next_actions自己单独处理要的索引
        # 使用目标网络来评估上面选择的动作的 Q 值

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # todo below is dqn
        # states, actions, rewards, next_states, dones = experiences
        # Q_targets_next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)  # (b,1)
        # Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        # loss = F.mse_loss(Q_expected, Q_targets)


        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        #
        # states, actions, rewards, next_states, dones = experiences
        #
        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) # 转为2维的

        # # 在强化学习中，这通常用于从目标网络中获取价值估计，而不影响梯度更新。
        # # Compute Q targets for current states
        #
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #
        # Q_expected = self.qnetwork_local(states).gather(1, actions)  # 固定行号，确认列号 在batchsize上面收集
        #
        # # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        # # Minimize the loss
        # self.optimizer.zero_grad()
        # #在反向传播之前，需要将优化器中累积的梯度清零。这是因为 PyTorch 在默认情况下会累积梯度，如果不清零，梯度将会与前一个批次的梯度相加。
        # loss.backward() # 回传计算梯度
        # self.optimizer.step() # 这个是更新梯度
        #
        # # ------------------- update target network ------------------- #
        # #这里没有使用论文中的多少次更新一次
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  #队列
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done) #可以把e看做是一个类变量，更方便使用
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #todo 这里
        experiences = random.sample(self.memory, k=self.batch_size) # 抽样

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        # from_numpy 它的作用是将一个 NumPy 数组转换为一个 PyTorch 张量（Tensor）共享相同的内存空间 共享类型
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        #重载
        """Return the current size of internal memory."""
        return len(self.memory)