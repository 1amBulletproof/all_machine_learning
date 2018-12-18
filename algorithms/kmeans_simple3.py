'''Implementation and of K Means Clustering
- adapted from https://flothesof.github.io/k-means-numpy.html '''

import numpy as np
import pandas as pd

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def kmeans(X, k, max_iters=15):
	centroids = initialize_centroids(X, k)
	for i in range(0,max_iters):
		closest = closest_centroid(X, centroids)
		centroids = move_centroids(X, closest, centroids)
	return centroids, closest

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
