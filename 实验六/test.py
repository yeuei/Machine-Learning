import numpy as np
a = np.array([[1,2,3],[4, 5, 6], [7, 8, 9]])
b = np.arange(5,8).reshape(-1,1)
c = np.array([[10, 11, 12],[13,14,15]])
print(a)
print(b)
print(b * a)

