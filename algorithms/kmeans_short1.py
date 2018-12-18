'''Implementation and of K Means Clustering
- adapted from https://gist.github.com/bistaumanga/6023692 '''
import numpy as np

def kmeans(data, num_clusters, maxIters = 10 ):

    centroids = data[np.random.choice(np.arange(len(data)), num_clusters), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in data])
        # Move centroids step
        centroids = [data[C == k].mean(axis = 0) for k in range(num_clusters)]
	#TODO: needs a stop step when old centroids == new centroids
    return np.array(centroids) , C

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
