#  importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

###################### class for implementing Neural Networks #####################

class NeuralNetwork():

    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(- z))
        return sigmoid

    def sigmoid_gradient(self, z):
        # returns the gradient of sigmoid function
        sigmoid_fn = self.sigmoid(z)
        sigmoid_grad = sigmoid_fn * (1 - sigmoid_fn)
        return sigmoid_grad

    def hypothesis(self, X, theta):
        # hypothesis for neural network
        z = np.dot(X, theta)
        hypothesis = self.sigmoid(z)
        return hypothesis

    def into_classes(self, X, y, num_classes):
        # to convert target values into classes by implementing one hot encoding
        Y = np.zeros((np.shape(X)[0], num_classes), dtype=int)
        for i in range(num_classes):
            Y[:, i] = np.where(y[:, 0] == i, 1, 0)
        return Y

    def cost_function(self, a3, Theta1, Theta2, y10, lamda):
        # returns the  log loss cost function with regularization
        m = y10.shape[0]
        cross_entropy = -y10 * np.log(a3) - (1 - y10) * np.log(1 - a3)
        cost_fn = np.sum(cross_entropy)/m + (lamda / 2.0 * m) * (np.sum(Theta1[1:, :] ** 2) + np.sum(Theta2[1:, :] ** 2))
        return cost_fn

    def forward_prop(self, X, Theta1, Theta2):
        # implementing forward propagation to get activation units of each layer
        hidden_layer_size = 32
        m = np.shape(X)[0]
        # a1 is the 1st layer
        a1 = X
        # a2 is the hidden layer (layer 2)
        a2 = np.ones((m, hidden_layer_size + 1))
        a2[:, 1:] = self.hypothesis(X, Theta1)
        # a3 is the prediction layer (layer 3)
        a3 = self.hypothesis(a2, Theta2)
        return a1, a2, a3

    def back_prop(self, Theta1, Theta2, X, y, lamda):
        # implementing back propagation to compute the gradients for theta1 and theta2
        m = np.shape(X)[0]
        a1, a2, a3 = self.forward_prop(X, Theta1, Theta2)
        cost = self.cost_function(a3, Theta1, Theta2, y, lamda)
        # d3 is the error in layer 3
        d3 = a3 - y
        # d2 is the error in layer 2
        d2 = np.dot(d3, Theta2.T) * self.sigmoid_gradient(a2)
        # grad 1 is the gradient for theta1
        grad1 = (np.dot(a1.T, d2)) / m
        # grad 2 is the gradient for theta2
        grad2 = (np.dot(a2.T, d3)) / m

        return cost, grad1[:, 1:], grad2

    def gradientDescent(self, X, y, Theta1, Theta2, alpha, iterations, lamda, num_classes):
        cost_history = []
        y10 = self.into_classes(X, y, num_classes)
        # iterating over no of iterations
        for i in range(iterations):
            cost, grad1, grad2 = self.back_prop(Theta1, Theta2, X, y10, lamda)
            # updating the values of theta1 and theta2
            Theta1 = Theta1 - (alpha * grad1)
            Theta2 = Theta2 - (alpha * grad2)
            # appending the cost on each iteration to cost_history
            cost_history.append(cost)

        return Theta1, Theta2, cost_history

    def predicting(self,Theta1, Theta2, X):
        # a3 contain the respective probabilites of each classes
        a1, a2, a3 = self.forward_prop(X, Theta1, Theta2)
        prediction = np.zeros((X.shape[0], 1))
        # choosing the class label with maximum probability as our prediction
        prediction[:, 0] = np.argmax(a3, axis=1)
        return prediction

    def accuracy(self, prediction, Y):
        # calculate the accuracy achieved on predictions
        count = 0
        m = np.shape(Y)[0]
        for i in range(m):
            if prediction[i][0] == Y[i][0]:
                count += 1
        return (float(count)/m) * 100

    def plot(self, cost_history):
        # to plot cost function with iterations
        plt.grid(True)
        plt.plot(cost_history)
        plt.suptitle('cost on training data vs. iterations')
        plt.xlabel('no of iterations')
        plt.ylabel('cost_train')
        plt.show()

    def neural_network(self, X_train, Y_train, X_cv, Y_cv, X_test, Y_test,
                       alpha=0.9, lamda=0, iterations=5000, num_classes=10, hidden_layer_size=32):
        n = X_train.shape[1]
        #  initialising theta1 and theta2 with random values
        initial_Theta1 = np.random.randn(n, hidden_layer_size)
        initial_Theta2 = np.random.randn(hidden_layer_size + 1, num_classes)

        Theta1, Theta2, cost_history = self.gradientDescent(X_train, Y_train, initial_Theta1, initial_Theta2,
                                                            alpha, iterations, lamda, num_classes)
        pred_train = self.predicting(Theta1, Theta2, X_train)
        pred_cv = self.predicting(Theta1, Theta2, X_cv)
        pred_test = self.predicting(Theta1, Theta2, X_test)

        accuracy_train = self.accuracy(pred_train, Y_train)
        accuracy_cv = self.accuracy(pred_cv, Y_cv)
        accuracy_test = self.accuracy(pred_test, Y_test)

        self.plot(cost_history)
        print("accuracy_train : ", accuracy_train, "%")
        print("accuracy_cv : ", accuracy_cv, "%")
        print("accuracy_test : ", accuracy_test, "%")
