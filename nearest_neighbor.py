# This script contains a class that classifies each image in the test set according to the nearest neighbor of its PCA
# transform from the transformed training set

import numpy as np
import matplotlib.pyplot as plt
from PCA import PCA

# Exercise 6.1


class NN():

    def __init__(self, PCA_train, PCA_test):

        # Transformed sets must have same number of rows
        if PCA_train.shape[0] != PCA_test.shape[0]:
            raise ValueError("Transformed sets must have the same number of rows")

        self.k = PCA_train.shape[0]
        self.PCA_train = PCA_train
        self.PCA_test = PCA_test

    # Nearest neighbor method to classify the columns of the test set
    def classify(self):

        # instantiate empty list of labels corresponding to training set columns
        label_list = []

        # iterate through each column of the test set
        for j in self.PCA_test.T:
            # instantiate empty list of differences
            difference_list = []

            # iterate through each column of the training set
            for i in self.PCA_train.T:
                energy_diff = np.linalg.norm(j - i) ** 2
                difference_list.append(energy_diff)

            # define the label index as minimum value of differences
            label_ind = np.argmin(difference_list)

            label_list.append(label_ind)

        return label_list


if __name__ == "__main__":
    # test our classification method for different values of k
    k_list = [1, 5, 10, 20]

    # read in training set and test set and labels
    training_set = np.load("training_set.npy")
    test_set = np.load("test_set.npy")
    training_labels = np.load("training_labels.npy")
    test_labels = np.load("test_labels.npy")

    # instantiate empty list for accuracy of classification
    accuracy_list = []

    # iterate through each value of k and return the predicted labels
    for k in k_list:
        object = PCA(training_set, test_set, k)
        training_transform = object.training_transform()
        test_transform = object. test_transform()

        pred_labels = NN(training_transform, test_transform).classify()

        # determine accuracy of classifications
        # append a 1 if correct, append a 0 if incorrect
        classification_list = []

        for test_ind, training_ind in enumerate(pred_labels):
            if test_labels[test_ind] == training_labels[training_ind]:
                classification_list.append(1)
            else:
                classification_list.append(0)

        # classification accuracy is mean of classification list
        accuracy = np.mean(classification_list)

        accuracy_list.append(accuracy)

        print(f"The classification accuracy for {k} principal components is {accuracy}")

    # plot classification accuracy vs. k
    plt.plot(k_list, accuracy_list)
    plt.title("Classification Accuracy vs. k Principal Components")
    plt.xlabel("k")
    plt.ylabel("Classification Accuracy")
    plt.xticks(np.arange(0, 21, 5))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.show()

    # iterate through different values of k for different images to show
    # effect of k on facial recognition

    image_list = [0, 14, 19]

    for k in k_list:
        object = PCA(training_set, test_set, k)
        training_transform = object.training_transform()
        test_transform = object.test_transform()

        pred_labels = NN(training_transform, test_transform).classify()

        for i in image_list:
            plt.imshow(test_set[:, i].reshape(-1, 112).T, cmap='gray')
            plt.title(f"Test Image of Subject {int(i + 1)}")
            plt.show()

            plt.imshow(training_set[:, pred_labels[i]].reshape(-1, 112).T, cmap='gray')
            plt.title(f"Nearest Neighbor of Subject {int(i + 1)} for k = {k}")
            plt.show()














