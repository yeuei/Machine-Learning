import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
ind1 = np.array([0, 1, 2])
ind2 = np.array([0.0, 1.0, 2.0])
a[ind1, ind2] = 0
print(a)
print(np.arange(10))