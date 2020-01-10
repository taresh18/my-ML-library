# importing the required libraries
import numpy as np
import pandas as pd

class MLlibrary(object):  # parent class containing the common functions used in the algorithms

    def __init__(self, train_data_filename, test_data_filename):
        # initialiing the training and testing data file names
        self.train_data_filename = train_data_filename
        self.test_data_filename = test_data_filename

    def load_dataset(self):
        # creating dataframes of training and testing dataset
        train_dataset = pd.read_csv(self.train_data_filename)
        test_dataset = pd.read_csv(self.test_data_filename)
        train_and_test = [train_dataset, test_dataset]
        return train_and_test

    def add_bias_features(self, X):
        # add a column containg biased featues to the input array
        X_updated = np.ones((X.shape[0], X.shape[1] + 1))
        X_updated[:, 1:] = X
        return X_updated

    def creating_arrays(self):
        # divide the training and testing dataset into various numpy arrays
        train_and_test = self.load_dataset()
        train_matrix = np.array(train_and_test[0])
        test_matrix = np.array(train_and_test[1])
        # add a column of biased featues to the train features array
        train_features = self.add_bias_features(train_matrix[:, 1:])
        # add a column of biased featues to the test features array
        test_features = self.add_bias_features(test_matrix[:, 1:])

        train_targets = train_matrix[:, 0].reshape((train_features.shape[0], 1))
        test_targets = test_matrix[:, 0].reshape((test_features.shape[0], 1))
        return [train_features, test_features, train_targets, test_targets]

    def mean_normalization(self):
        # applying mean normalisation on features
        arrays = self.creating_arrays()
        arrays[0] = ((arrays[0] - np.mean(arrays[0])) / (np.max(arrays[0]) - np.min(arrays[0])))
        arrays[1] = ((arrays[1] - np.mean(arrays[1])) / (np.max(arrays[1]) - np.min(arrays[1])))
        # returns a list containing all the fatures and target arrays
        return arrays

