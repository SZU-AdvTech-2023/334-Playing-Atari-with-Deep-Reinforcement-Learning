import gym
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent


import matplotlib.pyplot as plt
import cv2
import time



env = gym.make('Breakout-v0') # 为了缩放窗口 ,render_mode='human'
state_size = env.observation_space.shape # (210, 160, 3) 210*160 3通道
action_size = env.action_space.n
print('Original state shape: ', state_size)
print('Number of actions: ', env.action_space.n)
#breakout-v0的action_space有4个动作，分别是0-NOOP,1-FIRE,2-RIGHT,3-LEFT。
# 之前一直以为环境默认发出小球供击打，其实发出小球需要智能体做出动作1-FIRE。
print(env.action_space)

agent = Agent((32, 4, 84, 84), action_size, seed=1)  # state size (batch_size, 4 frames, img_height, img_width)
TRAIN = False  # train or test flag


def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    # print((observation.shape))
    # print(123)
    # observation = np.array(observation)
    # print(type(observation))
    # print(observation.shape) #(210, 160, 3)
    data = cv2.resize(src = observation, dsize= (84, 84))
    # print(data.shape)#(84, 84, 3)
    x_t = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) #缩放转灰度图
    # x_t = cv2.cvtColor(cv2.resize(src = observation, dsize= (84, 84)), cv2.COLOR_BGR2GRAY) #缩放转灰度图
    # print(23)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY) #灰度图转二值图,提取特征
    # cv2.imshow('img', x_t)
    # # cv2.imshow('adaptiveThreshold', img)
    # cv2.waitKey(0) #0为任意键位终止
    return x_t


def init_state(processed_obs):
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)


def dqn(n_episodes=30000, max_t=40000, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode, maximum frames
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # max-t todo
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    # 进行了这么多轮游戏 迭代了30000轮
    for i_episode in range(1, n_episodes + 1):
        obs = env.reset()
        obs = pre_process(obs)
        state = init_state(obs)
        print(1)
        score = 0
        # 这里是最大的timestep，如果没结束自己结束。相当于是一局游戏
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            # last three frames and current frame as the next state 每
            next_state = np.stack((state[1], state[2], state[3], pre_process(next_state)), axis=0)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        with open("info-log.txt","a") as f:
            f.write('\tEpsilon now : {:.2f}\n'.format(eps))
            f.write('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if i_episode % 1000 == 0:
                f.write('\n\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                f.write('\rEpisode {}\tThe length of replay buffer now: {}\n'.format(i_episode, len(agent.memory)))
            if np.mean(scores_window) >= 50.0:
                f.write('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                            np.mean(scores_window)))
                print("---------------------------------")

        print('\tEpsilon now : {:.2f}'.format(eps))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rEpisode {}\tThe length of replay buffer now: {}'.format(i_episode, len(agent.memory)))

        if np.mean(scores_window) >= 50.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                     np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint/dqn_checkpoint_solved.pth')
            break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint/dqn_checkpoint_8.pth')
    return scores


if __name__ == '__main__':
    # 0 - NOOP：不执行操作。
    # 1 - FIRE：开始游戏或发射球。
    # 2 - RIGHT：向右移动。
    # 3 - LEFT：向左移动。
    if TRAIN:
        start_time = time.time()
        scores = dqn()
        print('COST: {} min'.format((time.time() - start_time)/60))
        print("Max score:", np.max(scores))
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load('./dqn_checkpoint.pth'))
        rewards = []
        for i in range(10):  # episodes, play ten times
            total_reward = 0
            obs = env.reset()
            # print(type(obs))
            obs = pre_process(obs)
            state = init_state(obs)
            for j in range(10000):  # frames, in case stuck in one frame
                # s a r s'
                action = agent.act(state)
                env.render()
                next_state, reward, done, _ = env.step(action)
                # print(reward)
                state = np.stack((state[1], state[2], state[3], pre_process(next_state)), axis=0)
                total_reward += reward

                # time.sleep(0.01)
                if done:
                    rewards.append(total_reward)
                    break

        print("Test rewards are:", *rewards)
        print("Average reward:", np.mean(rewards))
        env.close()