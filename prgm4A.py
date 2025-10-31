import pandas as pd

df = pd.read_excel('ch1ex1.xlsx')
points = df.values

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(points)
labels = model.predict(points)

import matplotlib.pyplot as plt
xs = points[:,0]
ys = points[:,1]
plt.scatter(xs, ys, c=labels)
plt.show()


centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(xs, ys, c=labels)
plt.scatter(centroids_x, centroids_y, marker='X', s=200)
plt.show()
