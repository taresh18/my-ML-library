#  importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

################## class for implementing K means clustering ###############

class K_Means():

    def _init_(self, X, K):
        # initializing the frequently used values in the code
        self.n = X.shape[1]
        self.m = X.shape[0]
        self.K = K
        # initializing centroids
        self.centroids = self.initialize_centroids(X)

    def mean_normalizing(self, X):
        # applying mean normalization on input data
        normalized_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return normalized_X

    def initialize_centroids(self, X):
        # initialize centroids as k random features row from X
        init_centroids = X[np.random.randint(X.shape[0], size=self.K), :].reshape(self.K, self.n)
        return init_centroids

    def euclidean_distance(self, X_row):
        # calculate squared distance between each centroid and each features row
        X_row = X_row.reshape(1, self.n)
        # repeat a feature row k times to make it os same dimension as centroids
        X_row_matrix = np.repeat(X_row, repeats=self.K, axis=0)
        distance = np.sum(np.square(self.centroids - X_row_matrix), axis=1).reshape(1, self.n)
        # distance consists of the respective distances in separate columns
        return distance

    def assign_to_clusters(self, X):
        # assigning each point its nearest cluster
        cluster = np.zeros((self.m, 1))
        for row in range(self.m):
            cluster[row, 0] = np.argmin(self.euclidean_distance(X[row, :]), axis=1)
        return cluster

    def update_centroids(self, X, cluster):
        # updating centroids by taking mean of the updated clusters
        for n_centroid in range(self.K):
            X_rows = np.argwhere(n_centroid == cluster)
            X_rows = X_rows[:, 0]
            points = X[X_rows, :]
            self.centroids[n_centroid, :] = np.mean(points, axis=0)

    def training(self, iterations, X):
        # iterating and updating clusters and centroids
        for i in range(iterations):
            clusters = self.assign_to_clusters(X)
            self.update_centroids(X, clusters)
        return clusters

    def predicting(self, X_test):
        # clustering the test data
        m = X_test.shape[0]
        clusters = np.zeros((m, 1))
        for i in range(m):
            clusters[i, 0] = np.argmin(self.euclidean_distance(X_test[i]), axis=1)
        return clusters

    def plotX(self, X):
        # plotting features of X
        plt.grid(True)
        plt.suptitle("input data")
        plt.scatter(X[:, 0], X[:, 1])
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    def plotP(self, X, prediction):
        # plotting the predicted clusters of data
        plt.grid(True)
        plt.scatter(X[:, 0], X[:, 1], c = prediction)
        plt.suptitle("Predicted Clusters")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    def k_means(self, X_train, X_test, K):
        X_train = self.mean_normalizing(X_train)
        X_test = self.mean_normalizing(X_test)
        print("plotting X_train :")
        self.plotX(X_train)

        self._init_(X_train, K)
        prediction_train = self.training(K, X_train)
        prediction_train = prediction_train[:, 0]
        self.plotP(X_train, prediction_train)

        print("plotting X_test")
        self.plotX(X_test)
        prediction_test = self.predicting(X_test)
        prediction_test = prediction_test[:, 0]
        self.plotX(X_test, prediction_test)





