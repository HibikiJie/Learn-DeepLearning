from torch import nn
import torch
import gym
import random


class DQNet(nn.Module):

    def __init__(self, num_state, num_actions):
        """
        初始化网络
        :param num_state: 输入状态数
        :param num_actions: 输出行为数
        """
        super(DQNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, input_):
        return self.layer(input_)


class Trainer(object):

    def __init__(self, experience_pool_size, explore=0.8, foresight=0.9, is_cuda=True):
        self.device = torch.device('cuda:0' if is_cuda and torch.cuda.is_available() else 'cpu')

        """加载《倒杆杆游戏》"""
        self.game = gym.make('CartPole-v1')

        """定义网络远系数、探索系数、经验池、经验池尺寸"""
        self.foresight = foresight
        self.explore = explore
        self.experience_pool = []
        self.experience_pool_size = experience_pool_size

        """实例化网络、损失函数、优化器"""
        self.q_net = DQNet(num_state=4, num_actions=2).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters())
        self.loss_func = nn.MSELoss().to(self.device)

    def __call__(self):
        self.is_render = False  # 控制是否展示游戏画面
        """主循环"""
        self.ave = 0
        while True:
            self.__try_play()
            self.__train()

    def __try_play(self):
        """游戏初始化，并获取初始状态。这里是开始一局游戏。"""
        state_t = self.game.reset()
        R = 0
        """游戏循环，循环游戏的每一帧"""
        while True:

            """如果要展示游戏画面，则展示"""
            if self.is_render:
                self.game.render()

            """
            如果经验池没满，则先随机动作，充分试错，添加经验；
            否则，则通过网络通过预测的价值来选择动作。
            """
            if len(self.experience_pool) <= self.experience_pool_size:
                action = self.game.action_space.sample()

            else:
                """弹出一条经验值，"""
                self.experience_pool.pop(0)
                self.explore += 1e-4

                """随机选择，看是否需要探索，否则由网络输出动作"""
                if random.random() > self.explore:
                    action = self.game.action_space.sample()
                else:
                    _state_t = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.q_net(_state_t.to(self.device))
                    """根据最大价值，选择相应操作动作"""
                    action = torch.argmax(action.squeeze(0)).item()

            """执行动作，获取回报和下一状态"""
            state_t_plus_1, reward, done, info = self.game.step(action)

            """累加回报"""
            R += reward

            """向经验池增加经验"""
            self.experience_pool.append([state_t, R, action, state_t_plus_1, done])
            state_t = state_t_plus_1
            if done:
                self.ave = 0.95 * self.ave + 0.05 * R
                print(self.ave, R)
                if self.ave >= 300:
                    self.is_render = True
                    torch.save(self.q_net.state_dict(), 'q_net_CartPole.pth')
                break

    def __train(self):
        """训练过程"""
        if len(self.experience_pool) > self.experience_pool_size:
            """从经验池、获取经验"""
            states, values, actions, states_next, dons = self.get_experience()

            """得到当前状态的估计值"""
            valuations = self.q_net(states)
            valuation = torch.gather(valuations, dim=1, index=actions)

            """得到下一状态的估计值，并选择最大的估值"""
            valuations_next = self.q_net(states_next).detach()
            valuations_max = torch.max(valuations_next, dim=1, keepdim=True)[0]

            """当前状态的目标估值按照Q函数计算"""
            target = values + (1 - dons) * valuations_max * self.foresight

            """计算损失、优化"""
            loss = self.loss_func(valuation, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_experience(self):
        experiences = random.choices(self.experience_pool, k=100)
        states = []
        values = []
        actions = []
        states_next = []
        dons = []
        for experience in experiences:
            states.append(experience[0])
            values.append([experience[1]])
            actions.append([experience[2]])
            states_next.append(experience[3])
            dons.append([experience[4]])
        states = torch.tensor(states).float().to(self.device)
        values = torch.tensor(values).float().to(self.device)
        actions = torch.tensor(actions).to(self.device)
        states_next = torch.tensor(states_next).float().to(self.device)
        dons = torch.tensor(dons).float().to(self.device)
        return states, values, actions, states_next, dons

    def play(self):
        self.q_net.load_state_dict(torch.load('q_net_CartPole.pth'))
        for i_episode in range(20):

            """游戏初始化"""
            observation = self.game.reset()
            self.game.render()
            while True:
                """显示游戏"""
                self.game.render()

                """获取动作"""
                action = self.q_net(torch.tensor(observation).float().unsqueeze(0).to(self.device))
                action = torch.argmax(action.squeeze(0)).item()

                """动作一步，返回环境，回报，游戏是否结束，信息"""
                observation_, reward, done, info = self.game.step(action)
                observation = observation_
                # print(observation, action, reward, observation_, done)
                if done:
                    print('game over', )
                    break


if __name__ == '__main__':
    trainer = Trainer(10000)
    trainer.play()
