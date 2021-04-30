# This script reads in all of the images and creates a training set and testing set

import numpy as np
import matplotlib.pyplot as plt

# Exercise 4.1

# instantiate emtpy matrices for sets
training_set = np.zeros((10304, 360))   # 9 vectorized images of each of the 40 people
test_set = np.zeros((10304, 40))        # 1 vectorized image of each of the 40 people

# instantiate empty lists for labels
training_labels = np.zeros(360)
test_labels = np.zeros(40)

for i in range(1, 41):

    # reads in folder for each person
    person = "/Users/noahhschwab/PycharmProjects/ESE224_Lab11/att_faces/s" + str(i) + "/"

    # iterate through the first 9 images of a given person
    for j in range(1, 10):
        # define a given image
        image = plt.imread(person + str(j) + ".pgm")

        # vectorize the given image column-wise
        column = np.ravel(image, order='F')

        # append the vectorized image to the training set at the correct column index
        training_set[:, (i - 1) * 9 + j - 1] = column
        # append the label of the image to the label list
        training_labels[(i - 1) * 9 + j - 1] = int(i)

    # repeat the same process as above but only for the 10th image of each person for
    # the test set
    test_image = plt.imread(person + str(10) + ".pgm")
    test_column = np.ravel(test_image, order='F')
    test_set[:, i - 1] = test_column
    test_labels[i - 1] = int(i)

# save the sets as numpy files
np.save('training_set', training_set)
np.save('test_set', test_set)
np.save('training_labels', training_labels)
np.save('test_labels', test_labels)












