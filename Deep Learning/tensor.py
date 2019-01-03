import numpy as np
x = np.random.randint(1,100,50)
y = np.max(50*np.random.randn(50),0)
print(np.sin(x))
print(y)
x = [1,2,3,4]
y = [5,6,7,8]
x = np.reshape(x, (2, 2))
y = np.reshape(y, (2, 2))
z = np.dot(x.T, y)
print(np.shape(z), np.ndim(z))
