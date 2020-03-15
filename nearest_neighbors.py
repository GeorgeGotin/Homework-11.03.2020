from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dataset = load_iris()
X = dataset["data"]
y = dataset["target"]

def nearest_neighbors_classifier(X, y, x, k):
	minimal_index = [i for i in range(k)]
	minimal_dist = [calc_distance(x,X[i]) for i in minimal_index]
	for i in range(k,len(X)):
		dist = calc_distance(x, X[i])
		if dist < max(minimal_dist):
			minimal_index[minimal_dist.index(max(minimal_dist))] = i
			minimal_dist[minimal_dist.index(max(minimal_dist))] = dist
	res = []
	minimal_index.sort()
	for i in range(k):
		res.append(y[minimal_index[i]])
	return res
	


def calc_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k=5
hits = 0
for i in range(len(X_test)-k):
	x = X_test[i]
	y_true = y_test[i:k+i]  
	y_pred = nearest_neighbors_classifier(X_train, y_train, x, k)
	for j in range(k):
		if y_pred[j] == y_true[j]:
			hits += 1

print("{} out of {} are correct".format(hits, len(X_test)*k))
