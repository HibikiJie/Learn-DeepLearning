import torch

width = 6
height = 6
n = 4

board = torch.zeros(width, height)

def action_to_location(v):
	x = v % width
	y = v // width
	return y, x

def generate_indexes(v):
	ret = []
	x = v % width
	y = v // width
	# row
	if x <= width - n:
		ret.append(torch.tensor([z for z in range(v, v + n)]))
	# col
	if y <= height - n:
		ret.append(torch.tensor([z for z in range(v, v + n * width, width)]))
	# diagonal
	if x <= width - n and y <= height - n:
		ret.append(torch.tensor([z for z in range(v, v + n * (width + 1), width + 1)]))
	# anti diagonal
	if x >= n - 1 and y <= height - n:
		ret.append(torch.tensor([z for z in range(v, v + n * (width - 1), width - 1)]))

	return torch.stack(ret, dim=0)

def validate_line(board, v):
	indexes = generate_indexes(v)

	print(indexes, "indexes")
	if len(indexes) == 0:
		return
	t = board.take(indexes)
	print(t)
	t = torch.prod(t, dim=1)
	t = torch.sum(t, dim=0)
	print(t)
	if t > 0:
		print("ok")


print(generate_indexes(24))

board[action_to_location(8)] = 1
board[action_to_location(9)] = 1
board[action_to_location(10)] = 1
board[action_to_location(11)] = 1
board[action_to_location(14)] = 1
board[action_to_location(20)] = 1
board[action_to_location(26)] = 1
board[action_to_location(19)] = 1
board[action_to_location(24)] = 1
board[action_to_location(21)] = 1
board[action_to_location(22)] = 1
print(board)
validate_line(board, 0)
