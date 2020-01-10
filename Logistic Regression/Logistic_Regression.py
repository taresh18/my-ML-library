#  importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

##################### class for implementing Logistic Regression ######################

class LogisticRegression():

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

    def sigmoid(self, z):
        # return the sigmoid value of z
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def hypothesis(self, X, theta):
        # hypothesis for logistic regression
        z = np.dot(X, theta)
        hypothesis = self.sigmoid(z)
        return hypothesis

    def CostFunction(self, X, Y, theta, lamda):
        # returns the  log loss cost function with regularization
        m = Y.shape[0]
        log_loss = (- Y * np.log(self.hypothesis(X, theta))) - ((1 - Y) * np.log(1 - self.hypothesis(X, theta)))
        cost_function = (1 / m) * np.sum(log_loss) + (lamda / (2 * m)) * np.sum(np.square(theta[1:]))
        return cost_function

    def gradient(self, X, Y, theta, lamda):
        # returns gradient to be used in updating parameters
        m = Y.shape[0]
        # not including regularization term with biased parameters
        j_0 = 1 / m * np.dot(X.T, (self.hypothesis(X, theta) - Y))[0]
        j_1 = 1 / m * np.dot(X.T, (self.hypothesis(X, theta) - Y))[1:] + (lamda / m) * theta[1:]
        gradient = np.vstack((j_0[:, np.newaxis], j_1))
        return gradient

    def GradientDescent(self, X, Y, theta, alpha, iterations, lamda, batch_size):
        m = Y.shape[0]
        # cost_history contains the cost on training data with iterations
        cost_history = np.zeros((((X.shape[0] // batch_size) + 2) * iterations, 1))
        i = 0
        # iterate over the passed no of iterations
        for iteration in range(iterations):
            # implementing mini batch gradient descent for training
            mini_batches = self.create_mini_batches(X, Y, batch_size)
            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch
                cost = self.CostFunction(X_mini, Y_mini, theta, lamda)
                gradient = self.gradient(X_mini, Y_mini, theta, lamda)
                theta = theta - (alpha * gradient)
                cost_history[i, 0] = cost
                i += 1
        # returns the updated set of parameters and the list of costs per iteration
        return theta, cost_history

    def classifier(self, X, y, alpha, iterations, num_classes, lamda, batch_size):
        # implementing one Vs. all classification
        m, n = X.shape[0], X.shape[1]
        initial_theta = np.zeros((n, 1))
        # all_theta consists of a list of parameters for each class
        all_theta = []
        # all cost consists the cost history with iterations of each class in a separate column
        all_costs = np.zeros((((X.shape[0] // batch_size) + 2) * iterations, num_classes))

        for i in range(num_classes):
            # convert target values into classes by one-hot encoding
            y_into_classes = np.where(y == i, 1, 0)
            # find parameters and cost history for each individual class
            theta, cost_history = self.GradientDescent(X, y_into_classes, initial_theta, alpha, iterations, lamda, batch_size)
            all_theta.extend(theta)
            # putting cost history of individual classes into separate columns of all_J
            all_costs[:, [i]] = cost_history
        return np.array(all_theta).reshape(num_classes, n), all_costs

    def predicting(self, all_theta, X):
        m = X.shape[0]
        probabilities = np.dot(X, all_theta.T)
        # predictions equals to  the index(class) of probabilities array having the maximum probability
        predictions = np.argmax(probabilities, axis=1)
        predictions = np.array([predictions])
        predictions = predictions.T
        return predictions

    def plot(self, cost_history, ith_class):
        # to plot cost function with iterations
        cost_history = list(cost_history)
        plt.suptitle('cost_history of class label : ' + str(ith_class) + ' vs. iterations')
        plt.plot(cost_history)
        plt.grid(True)
        plt.xlabel('no of iterations')
        plt.ylabel('cost_train')
        plt.show()

    def accuracy(self, prediction, Y):
        # to calculate accuracy on predictions
        m = Y.shape[0]
        count = 0
        for i in range(np.shape(Y)[0]):
            if (prediction[i][0] == Y[i][0]):
                count += 1
        accuracy = float(count) / m
        return accuracy * 100

    def logistic_regression(self, X_train, Y_train, X_cv, Y_cv, X_test, Y_test,
                            alpha=0.3, lamda =30, iterations=100, num_classes=10, ith_class=1, batch_size=1024):
        all_theta, all_costs = self.classifier(X_train, Y_train, alpha, iterations, num_classes, lamda, batch_size)
        cost_history = all_costs[:, ith_class]

        pred_train = self.predicting(all_theta, X_train)
        pred_cv = self.predicting(all_theta, X_cv)
        pred_test = self.predicting(all_theta, X_test)

        accuracy_train = self.accuracy(pred_train, Y_train)
        accuracy_cv = self.accuracy(pred_cv, Y_cv)
        accuracy_test = self.accuracy(pred_test, Y_test)

        self.plot(cost_history, ith_class)
        print("accuracy_train :", accuracy_train, "%")
        print("accuracy_cv :", accuracy_cv, "%")
        print("accuracy_test :", accuracy_test, "%")
