'''Implementation and of K Means Clustering
- adapted from https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-7/ '''
import numpy as np
import pandas as pd

def find_closest_centroids(X, centroids):  
	m = X.shape[0]
	k = centroids.shape[0]
	idx = np.zeros(m)

	for i in range(m):
		min_dist = 1000000
		for j in range(k):
			dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
			if dist < min_dist:
				min_dist = dist
				idx[i] = j
	return idx

def compute_centroids(X, idx, k):
	m, n = X.shape
	centroids = np.zeros((k, n))

	for i in range(k):
		indices = np.where(idx == i)
		centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()

	return centroids

def init_centroids(X, k):  
	m, n = X.shape
	centroids = np.zeros((k, n))
	#print(centroids)
	#print(X)

	for i in range(k):
		idx = np.random.randint(0, n)
		#print(X[idx])
		centroids[i,:] = X[idx]
	#print(centroids)

	return centroids

def kmeans(X, k, max_iters=10):
	m, n = X.shape
	idx = np.zeros(m)
	centroids = init_centroids(X, k)

	for i in range(max_iters):
		idx = find_closest_centroids(X, centroids)
		centroids = compute_centroids(X, idx, k)

	return centroids, idx


def main():
	print('Main - test kmeans')
	input_data = pd.DataFrame([[1.0,1.1],[3.3,5.5],[1.2,0.9],[3.0,5.3]])
	print('Input data:')
	print(input_data)
	print('expected output is Centroids c1{1.1, 1.0} && c2{3.15, 5.4} & Clusters {1,3} && {2,4} ')
	num_clusters = 2
	centroids, clusters = kmeans(input_data.values, num_clusters)
	print('centroids')
	print(centroids)
	print('clusters')
	print(clusters)

	input_data = pd.DataFrame([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],[9,9,9,9]])
	print('Input data:')
	print(input_data)
	num_clusters = 3
	centroids, clusters = kmeans(input_data.values, num_clusters)
	print('centroids')
	print(centroids)
	print('clusters')
	print(clusters)

if __name__ == '__main__':
	main()
