import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x,y = make_blobs(n_samples=200,n_features=2,centers=1,cluster_std=0.5,shuffle=True)

import matplotlib.pyplot as plt
plt.scatter(x[:,0], x[:,1])
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300)
y_pred = km.fit_predict(x)

plt.scatter(x[y_pred==0,0], x[y_pred==0,1],label='cluster 1',marker='x')
plt.scatter(x[y_pred==1,0], x[y_pred==1,1],label='cluster 2',marker='^')
plt.scatter(x[y_pred==2,0], x[y_pred==2,0],label='cluster 3',marker='v')
plt.grid()
plt.show()