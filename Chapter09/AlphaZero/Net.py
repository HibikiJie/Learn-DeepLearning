import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self, board_size):
		super(Net, self).__init__()

		self.board_size = board_size

		# 综合网络
		self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

		# 策略网络，输出动作概率的对数 (方便后续计算)
		self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
		self.act_fc1 = nn.Linear(4*board_size*board_size, board_size*board_size)

		# 估值网络，输出当前局面价值 (归一化到-1 到 1之间）
		self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
		self.val_fc1 = nn.Linear(2*board_size*board_size, 64)
		self.val_fc2 = nn.Linear(64, 1)


	def forward(self, state_input):
		# common layers
		x = F.relu(self.conv1(state_input))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		# action policy layers
		x_act = F.relu(self.act_conv1(x))
		x_act = x_act.view(-1, 4*self.board_size*self.board_size)
		x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

		# state value layers
		x_val = F.relu(self.val_conv1(x))
		x_val = x_val.view(-1, 2*self.board_size*self.board_size)
		x_val = F.relu(self.val_fc1(x_val))
		x_val = torch.tanh(self.val_fc2(x_val))

		return x_act, x_val