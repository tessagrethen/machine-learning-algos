"""
Author: Tessa Pham
Description: Implements the k-nearest neighbors algorithm.
"""

import numpy as np
import math

def main():
    # load dataset into a numpy array
    train_data = np.loadtxt("zip.train")
    test_data = np.loadtxt("zip.test")

    # filter the data to only the rows with labels 2 or 3
    train_filtered = filter(train_data)
    test_filtered = filter(test_data)

    # calculate accuracy
    corrects = np.zeros(15)
    for test_e in test_filtered:
        for k in range(10):
            if knn(train_filtered, test_e[1:], k + 1) == test_e[0]:
                corrects[k] += 1
        # experiments
        if knn(train_filtered, test_e[1:], 25) == test_e[0]:
            corrects[10] += 1
        if knn(train_filtered, test_e[1:], 50) == test_e[0]:
            corrects[11] += 1
        if knn(train_filtered, test_e[1:], 75) == test_e[0]:
            corrects[12] += 1
        if knn(train_filtered, test_e[1:], 100) == test_e[0]:
            corrects[13] += 1

    num_test = len(test_filtered)
    print('Nearest Neighbors:')
    for k in range(10):
        percentage = (corrects[k] / num_test) * 100
        print('K = {}, {}%'.format(k + 1, round(percentage, 3)))
    # experiments
    percentage = (corrects[10] / num_test) * 100
    print('K = {}, {}%'.format(25, round(percentage, 3)))
    percentage = (corrects[11] / num_test) * 100
    print('K = {}, {}%'.format(50, round(percentage, 3)))
    percentage = (corrects[12] / num_test) * 100
    print('K = {}, {}%'.format(75, round(percentage, 3)))
    percentage = (corrects[13] / num_test) * 100
    print('K = {}, {}%'.format(100, round(percentage, 3)))

def filter(data):
    """Creates a new array that contains only rows with labels 2 or 3 (relabeled to -1 and 1)."""
    num_rows = len(data)
    indices = []

    for i in range(num_rows):
        if data[i][0] == 2:
            data[i][0] = -1
        elif data[i][0] == 3:
            data[i][0] = 1
        else:
            continue
        indices.append(i)

    return data[indices, :]

def knn(train, e, k):
    """Predicts label for the given example based on a nearest-neighbor classifier."""
    # each row in dists stores a distance and an index
    num_train = len(train)
    dists = np.empty([num_train, 2])

    # find distance between each training example and the test example
    for i in range(num_train):
        d = dist(train[i][1:], e)
        dists[i] = [d, i]

    # ascending sort for first column of dists
    dists = dists[np.argsort(dists[:, 0])]
    neighbor_dists = dists[:k]
    sign = 0
    for nd in neighbor_dists:
        n_index = nd[1].astype(int)
        neighbor = train[n_index]
        sign += neighbor[0]

    return 1 if sign > 0 else -1

def dist(a, b):
    """Calculates the distance between two examples."""
    num_features = len(a)
    sq_dist = 0

    for i in range(num_features):
        sq_dist += (a[i] - b[i])**2

    return math.sqrt(sq_dist)

if __name__ == "__main__":
    main()
