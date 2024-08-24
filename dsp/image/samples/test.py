import math
import numpy as np


def identity_filter(filter_size):
	array = [[0.0] * filter_size] * filter_size
	p = np.array(array, dtype=np.float32)
	i = math.floor(filter_size/2)
	p[i][i] = np.float32(1.0)
	print(p)


filter_size = 7
array = [[1] * filter_size] * filter_size
h = np.array(array, dtype=np.float32)
h = -(h / h.sum())
i = math.floor(filter_size/2)
h[i][i] = h[i][i] + 1
print(h)

