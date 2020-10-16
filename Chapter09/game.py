import gym

"""加载游戏"""
env = gym.make('MountainCar-v0')
for i_episode in range(20):

    """游戏初始化"""
    observation = env.reset()
    for t in range(100000):
        """显示游戏"""
        env.render()
        # print(observation)

        """从动作空间随机采样"""
        action = env.action_space.sample()
        """动作一步，返回环境，回报，游戏是否结束，信息"""
        observation_, reward, done, info = env.step(action)
        print(observation,action, reward, observation_, done)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
