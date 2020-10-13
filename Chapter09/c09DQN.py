from torch import nn
import torch
import gym


class DQNet(nn.Module):

    def __init__(self, num_state, num_actions):
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

    def __init__(self, experience_pool_size, explore=0.8):
        self.game = gym.make('CartPole-v1')
        self.q_net = DQNet(num_state=4, num_actions=2)
        self.optimizer = torch.optim.Adam(self.q_net.parameters())
        self.experience_pool = []
        self.experience_pool_size = experience_pool_size
        self.explore = explore
        self.loss_func = nn.MSELoss()

    def train(self):
        is_render = False
        while True:
            state_t = self.game.reset()
            while True:
                if is_render:
                    self.game.render()
                if len(self.experience_pool) <= self.experience_pool_size:
                    action = self.game.action_space.sample()
                else:
                    action = self.q_net(state_t)
