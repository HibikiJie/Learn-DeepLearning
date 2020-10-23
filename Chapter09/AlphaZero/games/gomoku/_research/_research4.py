import numpy as np

a = np.arange(12).reshape(2, 2, 3)

print(np.flip(a,axis=0))

print(np.swapaxes(a, 0, 1))