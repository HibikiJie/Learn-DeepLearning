import numpy as np
import random

"""初始化矩阵"""
Q = np.zeros((6, 6))

"""回报矩阵R"""
R = np.array([[-1, -1, -1, -1,  0,  -1],
              [-1, -1, -1,  0, -1, 100],
              [-1, -1, -1,  0, -1,  -1],
              [-1,  0,  0, -1,  0,  -1],
              [ 0, -1, -1,  0, -1, 100],
              [-1,  0, -1, -1,  0, 100]])

"""设立学习参数"""
gamma = 0.8

"""迭代"""
for i in range(20000):
    """对每一次迭代,随机选择一种状态作为初始"""
    state = random.randint(0, 5)
    while True:
        """选择当前状态下的所有可能动作"""
        actions = []
        for action in range(6):
            if R[state, action] >= 0:
                actions.append(action)

        """随机选择一个可以行动的动作"""
        state_next = actions[random.randint(0, len(actions) - 1)]

        """更新Q矩阵，通过当前状态，下一状态，该状态转移的回报，和该状态转移的价值。"""
        Q[state, state_next] = R[state, state_next] + gamma * (Q[state_next]).max()
        state = state_next
        """游戏最大可行动次数，达到即结束"""
        if state == 5:
            break

print((Q/Q.max()*100).astype(np.int))
