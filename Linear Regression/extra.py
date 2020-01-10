#  importing the required libraries
import numpy as np
import matplotlib.pyplot as plt


##########################   class for implementing Linear Regression algotithm   ############################

class LinearRegression():


    def create_mini_batches(self, X, Y, batch_size):
        # create mini batches from the input arrays of the given batch sizes
        mini_batches = []
        data = np.zeros((X.shape[0], X.shape[1] + 1))
        data[:, 1:] = X
        data[:, 0] = Y[:, 0]
        # n_minibatches = number of mini batches
        n_minibatches = data.shape[0] // batch_size

        i = 0
        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size : (i + 1) * batch_size, :]
            X_mini = mini_batch[:, 1:]
            Y_mini = mini_batch[:, 0].reshape((X_mini.shape[0], 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
           mini_batch = data[i * batch_size : data.shape[0]]
           X_mini = mini_batch[:, 1:]
           Y_mini = mini_batch[:, 0].reshape((X_mini.shape[0], 1))
           mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def hypothesis(self, X, theta):
        # hypothesis for linear regression
        hypothesis = np.dot(X, theta)
        return hypothesis

    def cost_function(self, X, Y, theta, lamda):
        # calculate the mean squred error on the trainig data with regularization
        hypothesis = self.hypothesis(X, theta)
        m = X.shape[0]
        mean_squared_error = np.square(hypothesis - Y)
        cost = (np.sum(mean_squared_error) + lamda * np.sum(theta[1:, :]))/(2.0 * m)
        return cost

    def gradient_descent(self, X, Y, theta, alpha, lamda):
        m = np.shape(X)[0]
        hypothesis = self.hypothesis(X, theta)
        # updating the values of parameters with regularization
        theta = theta - (alpha / m) * (np.dot(np.transpose(X), (hypothesis - Y)) + lamda * theta)
        return theta

    def training(self, X_train, Y_train, alpha, lamda, iterations, batch_size):
        cost_history = []
        theta = np.zeros((np.shape(X_train)[1], 1))
        # iterate over the passed no of iterations
        for iteration in range(iterations):
            # implementing mini batch gradient descent for training
            mini_batches = self.create_mini_batches(X_train, Y_train, batch_size)
            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch
                theta = self.gradient_descent(X_mini, Y_mini, theta, alpha, lamda)
                cost = self.cost_function(X_mini, Y_mini, theta, lamda)
                cost_history.append(cost)
        # return the updated set of parametes and the list of costs per iteration
        return theta, cost_history

    def predicting(self, X, theta):
        prediction = self.hypothesis(X, theta)
        prediction = np.round(prediction, 0)  # round off the prediction value to its nearest integer
        return prediction

    def plots(self, cost_history):
        # to plot cost history with iterations
        plt.suptitle('cost on training data vs. iterations')
        plt.plot(cost_history)
        plt.grid(True)
        plt.xlabel('no of iterations')
        plt.ylabel('cost_train')
        plt.show()

    def accuracy(self, prediction, Y):
        # to calculate the accuracy on predictions
        m = Y.shape[0]
        count = 0
        for i in range(np.shape(Y)[0]):
            if (prediction[i][0] == Y[i][0]):
              count += 1
        accuracy = (float(count) / m) * 100
        return round(accuracy, 2)  # rounding off the accuracy to 2 decimal places and returning it