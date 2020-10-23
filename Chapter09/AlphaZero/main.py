import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from Chapter09.AlphaZero.games.gomoku.game import Board, Game
from Chapter09.AlphaZero.mcts import MCTSPlayer
from Chapter09.AlphaZero.Net import Net


class PipeLine():

    def __init__(self, board_size=9, n=5):
        # 游戏棋盘大小
        self.board_size = board_size
        # 连五子胜利
        self.n = n
        # 游戏棋盘实例化
        self.board = Board(board_size=board_size, n=n)
        # 游戏实例化
        self.game = Game(self.board)
        # 学习率
        self.lr = 2e-3
        # 训练设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 每局自我训练模拟到终局次数
        self.n_playout = 500
        # 模拟时探索水平常数
        self.c_puct = 5
        # 记忆库大小
        self.buffer_size = 20000
        # 记忆库实例化
        self.data_buffer = deque(maxlen=self.buffer_size)
        # 批次
        self.batch_size = 512
        # 自我对局 1 次后进行训练
        self.n_games = 1
        # 自我对局后进行 5 次训练
        self.epochs = 5
        # 打印保存模型间隔
        self.check_freq = 50
        # 总共游戏次数
        self.game_num = 10000
        # 模型实例化
        self.model = Net(board_size).to(self.device)
        # 优化器 (带有权重衰减防止过拟合)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-4)
        # 实例化蒙特卡洛玩家，参数：游戏策略，探索常数，模拟次数，是否自我对弈（测试时为False）
        self.mcts_player = MCTSPlayer(policy=self.policy,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_self_play=True)

    def run(self):
        # 启动主流程，循环自我对抗训练
        for i in range(self.game_num):
            # 自我对弈
            self.self_play(self.n_games)
            print("self ... play ... {}".format(i))

            # 如果记忆库大小大于批次则开始训练
            if len(self.data_buffer) > self.batch_size:
                loss = self.policy_update()
                print("loss ... {}".format(loss.item()))

            # 保存模型
            if (i + 1) % self.check_freq == 0:
                self.save("models/model.pth")

    def policy(self, board):
        # 获取可用动作
        avail_actions = board.avail_actions
        state = torch.from_numpy(board.observe).unsqueeze(0).to(self.device)

        # 根据状态通过模型获取 log 动作概率和价值
        log_act_probs, value = self.model(state)

        # 把 log 动作概率转换为动作概率并过滤不可用动作
        act_probs = torch.exp(log_act_probs).detach().cpu().numpy().flatten()
        act_probs = zip(avail_actions, act_probs[avail_actions])
        value = value.item()

        # 返回动作概率，当前局面价值
        return act_probs, value

    def self_play(self, n_games=1):
        for i in range(n_games):
            winner, data = self.game.self_play(self.mcts_player, temp=1.0)
            print(data)
            # 打印每局对局信息
            print(self.game.board, "\n", "------------------xx--------")
            # 获取扩展数据
            data = self.get_extend_data(data)
            # 存入记忆库
            self.data_buffer.extend(data)

    def get_extend_data(self, data):
        extend_data = []
        for state, mcts_porb, winner in data:
            extend_data.append((state, mcts_porb, winner))
            # 分别旋转 90度/180度/270度
            for i in range(1, 4):
                # 同时旋转棋盘状态和概率值
                _state = np.rot90(state, i, (1, 2))
                _mcts_prob = np.rot90(mcts_porb.reshape(self.board.height, self.board.width), i)
                extend_data.append((_state, _mcts_prob.flatten(), winner))

                # 翻转棋盘
                _state = np.array([np.fliplr(s) for s in _state])
                _mcts_prob = np.fliplr(_mcts_prob)
                extend_data.append((_state, _mcts_prob.flatten(), winner))
        return extend_data

    def policy_update(self):
        # 采样
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # 解包
        state_batch, mcts_probs_batch, winner_batch = zip(*mini_batch)  # tuple<np>, tuple<np>, tuple<np.float>

        loss = 0.
        for i in range(self.epochs):
            loss = self.train(state_batch, mcts_probs_batch, winner_batch)

        return loss

    def train(self, state_batch, mcts_probs, winner_batch):
        # 数据处理
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)

        # 回顾记忆状态通过模型输出动作值，价值
        log_act_probs, value = self.model(state_batch)

        # 计算损失
        # 价值损失：输出价值与该状态所在对局最终胜负的值（-1/0/1）（均方差）
        # 策略损失：蒙特卡洛树模拟的概率值与神经网络模拟的概率值的相似度 (-log(pi) * p)
        value_loss = F.mse_loss(value, winner_batch.view_as(value))
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    pipeLine = PipeLine()
    pipeLine.run()
