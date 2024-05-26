import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

x = np.arange(-30, 30).reshape(1, -1)

a = 2
b = 4
c = 6

k = (a**2 + b**2)**0.5
n1 = a/k
n2 = b/k
d = c/k

l = np.array([[n1], [n2], [d]])

'''
n1*x + n2*y + d = 0
y = (-d -n1*x)/n2
'''

y = (-d-n1*x)/n2
noise = np.random.normal(0, 5, x.shape)

y_noise = y +  noise

# Obtaining centroid
centroid_x = np.average(x)
centroid_y = np.average(y_noise)

# Convariance of the data and PCA
data = np.concatenate((x, y_noise), axis=0)
cov_matrix = np.cov(data, bias=True)

w,v=eig(cov_matrix)

idx = np.argmax(w)

V = v[:, idx]
W = w[idx]

y_predicted = (V[1]/V[0])*(x - centroid_x) + centroid_y

dist = np.sum((x - y_predicted)**2)
print('Minimized square error: ', dist)

plt.scatter(x, y_noise)
plt.scatter(x, y_predicted, color='red', label='Predicted line')
plt.grid()
# plt.xlim((-35, 35))
# plt.ylim((np.min(y)-5, np.max(y)+5))
plt.show()