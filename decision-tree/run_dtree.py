"""
Main program to run decision tree model on train and test datasets.
Author: Tessa Pham
Date: 09/16/2019
"""

import util
from decimal import *
from DecisionTree import *

def main():
    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename, True)
    test_partition  = util.read_arff(opts.test_filename, False)

    # create a DecisionTree instance from training data
    if opts.depth:
        DecisionTree.max_depth = opts.depth

    train_dtree = DecisionTree(train_partition, 0)

    # print text representation of the decision tree
    print(train_dtree)

    # evaluate the decision tree on test data
    correct = 0
    for e in test_partition.data:
        if train_dtree.predict(e) == e.label:
            correct += 1

    print(f'{correct} out of {test_partition.n} correct')
    accuracy = Decimal(f'{correct / test_partition.n}').quantize(Decimal('1.0000'))
    print(f'accuracy: {accuracy}')

main()
