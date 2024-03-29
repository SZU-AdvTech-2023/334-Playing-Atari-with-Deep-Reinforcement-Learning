import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gym
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 用于提取特征
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) #这个是确定cpu上tensor的随机数
        "*** YOUR CODE HERE ***"
        print(state_size) # torch.Size([32, 4, 84, 84]) batch size = 32 b*c*w*h
        self.conv = nn.Sequential(
            #这里的和论文的有些不一样
            nn.Conv2d(state_size[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512), # 32*64*7*7
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        conv_out = self.conv(state).view(state.size()[0], -1) #保留
        return self.fc(conv_out)


def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(x_t, (1, 84, 84)), x_t


def stack_state(processed_obs):
    """Four frames as a state"""
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)


if __name__ == '__main__':

    env = gym.make('Breakout-v0')
    # print('State shape: ', env.observation_space.shape)
    # print('Number of actions: ', env.action_space.n)

    obs = env.reset() # 210*160*3
    x_t, img = pre_process(obs) # 84*84
    state = stack_state(img) #变成4帧
    # print(state)
    # print(np.shape(state[0]),"123") # 4*84*84

    # plt.imshow(img, cmap='gray')
    # 用cv2模块显示
    # cv2.imshow('Breakout', img)
    # cv2.waitKey(0)

    state = torch.randn(32, 4, 84, 84)  # (batch_size, color_channel, img_height,img_width)
    state_size = state.size()

    cnn_model = QNetwork(state_size, action_size=4, seed=1)
    # cnn_model.to(device=device)
    outputs = cnn_model(state)
    # print(outputs)
