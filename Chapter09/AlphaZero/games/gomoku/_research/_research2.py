import numpy as np

width = 6
height = 6
n = 4

board = np.zeros((width, height))

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
		ret.append(np.array([z for z in range(v, v + n)]))
	# col
	if y <= height - n:
		ret.append(np.array([z for z in range(v, v + n * width, width)]))
	# diagonal
	if x <= width - n and y <= height - n:
		ret.append(np.array([z for z in range(v, v + n * (width + 1), width + 1)]))
	# anti diagonal
	if x >= n - 1 and y <= height - n:
		ret.append(np.array([z for z in range(v, v + n * (width - 1), width - 1)]))

	return np.array(ret)

def validate_line(board, v):
	indexes = generate_indexes(v)

	print(indexes, "indexes")
	if len(indexes) == 0:
		return
	t = board.take(indexes)
	print(t)
	t = np.prod(t, axis=1)
	t = np.sum(t)
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
board[action_to_location(28)] = 1
board[action_to_location(35)] = 1
print(board)
validate_line(board, 14)

print("===============================")

a = np.zeros((4, 4))
a[1,1] = 1
a[2,1] = 1
a[3,0] = 1
print(a)
print(a.flatten())
print(a.flatten().nonzero()[0])

