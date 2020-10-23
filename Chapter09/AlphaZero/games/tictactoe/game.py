import numpy as np
import torch

class Board():

	def __init__(self):

		self.width = 3
		self.height = 3
		self.size = (self.height, self.width)
		self.players = [0, 1]
		self.init_state()

	def init_state(self, start_player=0):

		self.current_player = start_player
		self.avail_actions = list(range(self.size[0] * self.size[1]))
		self.state = np.zeros((2, *self.size), dtype=np.float32)
		self.last_action = -1

	def action_to_location(self, action):
		h = action // self.size[0]
		w = action % self.size[1]
		return [h, w]

	def location_to_action(self, location):
		assert len(location) == 2, 'Wrong Location'
		h = location[0]
		w = location[1]
		action = h * self.size[0] + w
		assert action in range(self.size[0] * self.size[1]), 'Wrong Location'
		return action

	def step(self, action):
		location = self.action_to_location(action)
		self.state[self.current_player][location[0]][location[1]] = 1.
		self.avail_actions.remove(action)
		self.last_action = action
		self.current_player = 1 - self.current_player

		return self.end

	def check_win(self):
		for player in self.players:
			_state = self.state[player]
			# row
			if 3 in np.sum(_state, axis=1):
				return True
			# col
			elif 3 in np.sum(_state, axis=0):
				return True
			# bevel-1
			elif np.sum(_state * np.eye(_state.shape[0])) == 3:
				return True
			# bevel_2
			elif np.sum(_state * np.flip(np.eye(_state.shape[0]), axis=0)) == 3:
				return True
		return False

	@property
	def end(self):
		if self.check_win():
			return True, 1 - self.current_player
		elif not len(self.avail_actions):
			return True, -1
		return False, -1

	@property
	def observe(self):
		cp = np.zeros((1, self.height, self.width)) + self.current_player
		obs = np.concatenate((self.state, cp), axis=0).astype(np.float32)
		return obs

	def __str__(self):
		ret = ""
		for i in range(3):
			row = ""
			for j in range(3):
				if self.state[0, i, j] == 1:
					row += "o"
				elif self.state[1, i, j] == 1:
					row += "x"
				else:
					row += "-"
			row += "\n"
			ret += row
		return ret


class Game():

	def __init__(self, board):
		self.board = board

	def play(self, p1, p2, start_player=0, is_shown=False):
		players = (p1, p2)
		self.board.init_state(start_player)
		while True:
			if is_shown:
				print(self.board)
			current_player = players[self.board.current_player]

			action = current_player.get_action(self.board)
			terminal, winner = self.board.step(action)

			if terminal:
				if is_shown:
					print(self.board)
					if winner != -1:
						print("Game end. Winner is", players[winner])
					else:
						print("Game end. Tie")
				return winner

	def self_play(self, player, temp=1e-3, is_shown=False):
		self.board.init_state()
		states, mcts_probs, current_players = [], [], []
		while True:
			if is_shown:
				print(self.board)
			action, action_probs = player.get_action(self.board, temp=temp, return_prob=True)

			states.append(self.board.observe)
			mcts_probs.append(action_probs)
			current_players.append(self.board.current_player)

			terminal, winner = self.board.step(action)

			if terminal:
				# winner from the perspective of the current player of each state
				winners_z = np.zeros(len(current_players))
				if winner != -1:
					winners_z[np.array(current_players) == winner] = 1.0
					winners_z[np.array(current_players) != winner] = -1.0
				# reset MCTS root node
				player.reset_player()
				if is_shown:
					if winner != -1:
						print("Game end. Winner is player:", winner)
					else:
						print("Game end. Tie")

				return winner, list(zip(states, mcts_probs, winners_z))

######## TEST #######
if __name__ == "__main__":
	board = Board()
	board.init_state()
	game = Game(board)
	# board.step(0)
	# board.step(1)
	# board.step(2)
	# board.step(4)
	# board.step(5)
	# board.step(7)
	# print(board)
	# print(board.end)

	board.step(4)
	board.step(8)
	board.step(2)
	board.step(7)
	board.step(6)
	print(board)
	print(board.end)
