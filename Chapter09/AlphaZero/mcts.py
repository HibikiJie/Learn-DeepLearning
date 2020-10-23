import numpy as np
import copy

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs /= np.sum(probs)
	return probs

def rollout_policy_fn(board):
	action_probs = np.random.rand(len(board.avail_actions))
	return list(zip(board.avail_actions, action_probs))

# mcts 树节点
class TreeNode:
	def __init__(self, parent=None, prob=1.0):
		self.parent = parent
		self.children = {}

		# 该节点被访问次数
		self.n_visits = 0
		# 该节点的Q值（胜率）
		self.Q = 0
		# 该节点的u值（访问率）
		self.u = 0
		# 该节点被访问的概率
		self.P = prob

	@property
	def is_leaf(self):
		return self.children == {}

	@property
	def is_root(self):
		return self.parent is None

	# 扩展节点
	# 纯mcts: 平均概率扩展节点，神经网络：根据父节点选择的概率扩展节点
	def expand(self, action_probs):
		for action, prob in action_probs:
			if action not in self.children:
				self.children[action] = TreeNode(self, prob)

	# 选择价值最大的子节点
	def select(self, c_puct):
		return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

	# 根据最终胜负更新Q值
	def update(self, value):
		self.n_visits += 1
		self.Q += 1.0 * (value - self.Q) / self.n_visits

	# 回溯树节点更新每个节点的价值
	def backup(self, value):
		if self.parent:
			self.parent.backup(-value)
		self.update(value)

	# 获取价值，UCT（Upper Confidence Bound，上限置信区间算法）
	# Q = 胜利次数/模拟次数，c_puct = 探索常数， self.P = 被访问的概率， np.sqrt(self.parent.n_visits) / (1 + self.n_visits) = 父节点被访问次数/节点被访问次数
	def get_value(self, c_puct):
		self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
		return self.Q + self.u


class MCTS:

	def __init__(self, policy, c_puct, n_playout, pure):
		self.root = TreeNode()
		self.policy = policy
		self.c_puct = c_puct
		self.n_playout = n_playout
		self.pure = pure

	def get_action(self, state):
		for n in range(self.n_playout):
			_state = copy.deepcopy(state)
			self.playout(_state)
		return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]
	
	def get_action_probs(self, state, temp=1e-3):
		# 模拟（Simulation）
		for n in range(self.n_playout):
			state_copy = copy.deepcopy(state)
			self.playout(state_copy)

		# 根据节点被访问次数输出概率值
		act_visits = [(act, node.n_visits)  for act, node in self.root.children.items()]
		acts, visits = zip(*act_visits)
		act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

		return acts, act_probs

	def playout(self, state):
		node = self.root
		# 选择（Selection），找一个最好的值得探索的结点，通常是先选择没有探索过的结点，如果都探索过了，再选择UCB值最大的进行选择
		while not node.is_leaf:
			action, node = node.select(self.c_puct)
			state.step(action)

		# 通过策略获取动作概率值和价值
		# 纯mcts：输出平均概率和空价值
		# 神经网络mcts：输出估计概率值和当前局面价值
		action_probs, leaf_value = self.policy(state)

		# 获取当前局面是否终止，如果未终止则进行扩展 (Expansion），如果终止则用最终胜负替换神经网络估计的叶节点值
		terminal, winner = state.end

		if not terminal:
			node.expand(action_probs)
		else:
			if winner == -1:
				leaf_value = 0.
			else:
				leaf_value = 1. if winner == state.current_player else -1.
			
		# 如果使用纯粹mcts通过rollout模拟到终局来模拟当前局面价值
		if self.pure:
			leaf_value = self.rollout(state)

		# 回溯更新
		node.backup(-leaf_value)

	# 采用随机下子模拟到终局获取胜负数据
	def rollout(self, state, limit=1000):
		player = state.current_player
		for i in range(limit):
			terminal, winner = state.end
			if terminal:
				break
			action_probs = rollout_policy_fn(state)
			max_action = max(action_probs, key=lambda p: p[1])[0]

			state.step(max_action)
		else:
			assert False, "max limit rollout"
		if winner == -1:  # tie
			return 0
		else:
			return 1 if winner == player else -1
	
	def update_with_action(self, action):
		if action in self.root.children:
			self.root = self.root.children[action]
			self.root.parent = None
		else:
			self.root = TreeNode(None, 1.0)

#
class MCTSPlayer:

	def __init__(self, policy=None, c_puct=5, n_playout=2000, is_self_play=False, print_detail=False):

		# 判断纯粹mcts策略还是神经网路mcts策略
		if policy is None:
			self.pure = True
			policy = self.default_policy
		else:
			self.pure = False
		# 是否自我对弈
		self.is_self_play = is_self_play
		# 实例化mcts算法
		self.mcts = MCTS(policy, c_puct, n_playout, self.pure)
		# 是否打印AI每个动作概率值（测试时开启）
		self.print_detail = print_detail

	# 参数：棋盘/概率修正常数/是否同时返回概率
	def get_action(self, board, temp=1e-3, return_prob=False):
		# 获取当前状态动作
		if self.pure:
			return self._get_pure_action(board)
		else:
			return self._get_alpha_zero_action(board, temp, return_prob)

	def _get_pure_action(self, board):
		avail_actions = board.avail_actions
		assert len(avail_actions) > 0, "Full Board"

		action = self.mcts.get_action(board)
		self.mcts.update_with_action(-1)
		return action

	def _get_alpha_zero_action(self, board, temp, return_prob):
		avail_actions = board.avail_actions
		assert len(avail_actions) > 0, "Full Board"

		# 根据棋盘当前状态通过mcts算法获取动作和概率值
		action_probs = np.zeros(board.width * board.height, dtype=np.float32)
		acts, probs = self.mcts.get_action_probs(board, temp)
		action_probs[list(acts)] = probs
		if self.print_detail:
			self.print_probs(action_probs.reshape(board.width, board.height))

		if self.is_self_play:
			# 自我对弈训练时加入 狄利克雷分布 探索后根据概率选择动作
			action = np.random.choice(
				acts,
				p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
			)
			self.mcts.update_with_action(action)
		else:
			# 测试时直接选择根据概率选择动作
			action = np.random.choice(acts, p=probs)
			self.mcts.update_with_action(-1)

		if return_prob:
			return action, action_probs
		else:
			return action

	def default_policy(self, state):
		action_probs = np.ones(len(state.avail_actions)) / len(state.avail_actions)
		return zip(state.avail_actions, action_probs), 0

	def reset_player(self):
		self.mcts.update_with_action(-1)

	def print_probs(self, probs):
		for i in range(probs.shape[0]):
			stri = ""
			for j in range(probs.shape[1]):
				value = str(round(probs[i, j].item() * 100, 2))
				value = (" " * (6 - len(value))) + value
				stri += "{} % ".format(value)
			print(stri)
		print("----------------------------")


