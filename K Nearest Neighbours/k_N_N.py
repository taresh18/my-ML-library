#  importing the required libraries
import numpy as np

####################### class for implementing knn #####################

class KNN():

    def euclidean_distance(self, X_train, X_test, Y_train):
        # calculate the euclidean distance bewteen a test feature row and the train fatures
        m = X_train.shape[0]
        distances = np.zeros((m, 2))
        squared_distance = np.sum(np.square(X_train - X_test), axis=1).reshape(m, 1)
        # 0th column of distances stores the class label
        distances[:, 0] = Y_train[:, 0]
        # 1st column of distances stores the distance from respective classes
        distances[:, 1] = squared_distance[:, 0]
        return distances

    def finding_neighbours(self, X_train, Y_train, X_test_row, k):
        # here X_test is an array consisting of a row of test features duplicated over m times
        X_test = np.repeat(X_test_row, repeats=X_train.shape[0], axis=0)
        distances = self.euclidean_distance(X_train, X_test, Y_train)
        # now, we sort the distances array according to distances
        ind = np.argsort(distances[:, 1])
        distances = distances[ind]
        neighbours = np.zeros((k, 1))
        for i in range(k):
            # stores the first k class labels according to increasing distances from the test features row
            neighbours[i][0] = distances[i][0]
        return neighbours

    def predicting_class(self, X_train, Y_train, X_test_row, k):
        neighbours = self.finding_neighbours(X_train, Y_train, X_test_row, k)
        # we select the prediction based on the frquency of a class label in the neighbours array
        (values, counts) = np.unique(neighbours, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def finding_predictions(self, X_train, Y_train, X_test, k):
        m = X_test.shape[0]
        predictions = np.zeros((m, 1))
        # iterates over whole of the test array to predict the respective class labels
        for i in range(m):
            X_test_row = X_test[i, :].reshape(1, X_test.shape[1])
            predictions[i, 0] = self.predicting_class(X_train, Y_train, X_test_row, k)
        return predictions

    def accuracy(self, predictions, Y_test):
        # to calculate the accuracy of predictions
        m = np.shape(Y_test)[0]
        count = 0
        for i in range(m):
            if predictions[i][0] == Y_test[i][0]:
                count += 1
        accuracy = float(count) / m
        return accuracy * 100

    def KNN(self, X_train, X_test, Y_train, Y_test, k=5):
        # k is the number of nearest neighbours required
        predictions = self.finding_predictions(X_train, Y_train, X_test, k)
        accuracy = self.accuracy(predictions, Y_test)
        print("Accuracy achieved :", accuracy, "%")