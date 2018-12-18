'''Implementation and of K Means Clustering
- adapted from https://mubaris.com/2017/10/01/kmeans-clustering-in-python/'''
from copy import deepcopy
import numpy as np
import pandas as pd

# Euclidean Distance Caculator
def dist(a, b, ax=1):
	print('a and b')
	print(a)
	print(b)
	return np.linalg.norm(a - b, axis=ax)

def kmeans(data, k):
	# Number of clusters

	#Customization for multiple input
	centroids = data.sample(k)
	#print('centroids')
	#print(centroids)

	# X coordinates of random centroids
	#C_x = np.random.randint(0, np.max(X)-20, size=k)
	# Y coordinates of random centroids
	#C_y = np.random.randint(0, np.max(X)-20, size=k)
	#C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
	C = centroids.values
	X = data.values
	#print(C)

	# To store the value of centroids when it updates
	C_old = np.zeros(C.shape)

	# Cluster Lables(0, 1, 2)
	clusters = np.zeros(len(X))

	# Error func. - Distance between new centroids and old centroids
	error = dist(C, C_old, None)
	print('error')
	print(error)

	# Loop will run till the error becomes zero
	while error != 0:
		# Assigning each value to its closest cluster
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		# Storing the old centroid values
		C_old = deepcopy(C)

		# Finding the new centroids by taking the average value
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)

		error = dist(C, C_old, None)
		print('error2')
		print(error)

	return C, clusters


def main():
	print('Main - test kmeans')
	input_data = pd.DataFrame([[1.0,1.1],[3.3,5.5],[1.2,0.9],[3.0,5.3]])
	print('Input data:')
	print(input_data)
	print('expected output is Centroids c1{1.1, 1.0} && c2{3.15, 5.4} & Clusters {1,3} && {2,4} ')
	num_clusters = 2
	centroids, clusters = kmeans(input_data, num_clusters)
	print('centroids')
	print(centroids)
	print('clusters')
	print(clusters)

	input_data = pd.DataFrame([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],[9,9,9,9]])
	print('Input data:')
	print(input_data)
	num_clusters = 3
	centroids, clusters = kmeans(input_data, num_clusters)
	print('centroids')
	print(centroids)
	print('clusters')
	print(clusters)

if __name__ == '__main__':
	main()
