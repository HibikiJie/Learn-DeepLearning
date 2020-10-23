
import numpy as np
#
# a = np.arange(9).reshape(3, 3)
# prob = (a / np.sum(a)).flatten()
# print(a)
# print(prob.reshape(3,3))
#
# b = np.rot90(a, 1)
# print(b)
#
# prob_b = np.rot90(prob.reshape(3,3), 1).flatten()
# prob_bs = np.flipud(np.rot90(np.flipud(prob.reshape(3,3)), 1)).flatten()
# print(prob_b.reshape(3,3))
# print(prob_bs.reshape(3,3))
# 000000000000000000000000000000000000000000000000000000000

# c = np.rot90(a, 2)
# print(c)
#
# c = np.rot90(a, 3)
# print(c)
#
# c = np.rot90(a, 4)
# print(c)

# print(np.flipud(c))
# print(np.fliplr(c))
# print("------------------------------------------------------")
# state = np.arange(8).reshape(2, 2, 2)
# prob = (np.arange(4) / (0+1+2+3)).flatten()
# print(state, "\n ------", prob.reshape(2, 2))
#
# equi_state = np.array([np.rot90(s, 1) for s in state])
# equi_mcts_prob = np.rot90(np.flipud(prob.reshape(2, 2)), 1)
# print(equi_state,"\n -----", np.flipud(equi_mcts_prob),"\n -----")

# flip horizontally
# equi_state = np.array([np.fliplr(s) for s in equi_state])
# equi_mcts_prob = np.fliplr(equi_mcts_prob)
# extend_data.append((equi_state,
# 					np.flipud(equi_mcts_prob).flatten(),
# 					winner))

a = np.arange(8).reshape(2, 2, 2)
print(a, "===")
print(np.fliplr(a))

print(np.rot90(a, 1, (1, 2)))