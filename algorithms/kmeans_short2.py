'''Implementation and of K Means Clustering
- adapted from https://jonchar.net/notebooks/k-means/ '''

import numpy as np

np.random.seed(0)

def initialize_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(points.shape[0], size=k)]
    
# Function for calculating the distance between centroids
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)

def kmeans(X, k=2, maxiter = 50):
	# Initialize our centroids by picking random data points
	centroids = initialize_clusters(X, k)

	# Initialize the vectors in which we will store the
	# assigned classes of each data point and the
	# calculated distances from each centroid
	classes = np.zeros(X.shape[0], dtype=np.float64)
	distances = np.zeros([X.shape[0], k], dtype=np.float64)
		
	# Loop for the maximum number of iterations
	for i in range(maxiter):
	
		# Assign all points to the nearest centroid
		for i, c in enumerate(centroids):
			distances[:, i] = get_distances(c, X)
	
		# Determine class membership of each point
		# by picking the closest centroid
		clusters = np.argmin(distances, axis=1)
	
		# Update centroid location using the newly
		# assigned data point clusters
		for c in range(k):
			centroids[c] = np.mean(X[clusters == c], 0)

	#TODO need a step to stop iterating if centroids are unchanged
	return centroids, clusters


def main():
	print('Main - test kmeans')
	input_data = np.array([[1.0,1.1],[3.3,5.5],[1.2,0.9],[3.0,5.3]])
	print('Input data:')
	print(input_data)
	print('expected output is Centroids c1{1.1, 1.0} && c2{3.15, 5.4} & Clusters {1,3} && {2,4} ')
	num_clusters = 2
	centroids, clusters = kmeans(input_data, num_clusters)
	print('centroids')
	print(centroids)
	print('clusters')
	print(clusters)

	input_data = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],[9,9,9,9]])
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
