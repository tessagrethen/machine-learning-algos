"""
Author: Tessa Pham
Description: Stores data structure for decision tree.
Date: 09/12/2019
"""

import copy
from Partition import *

class DecisionTree:

    max_depth = None

    def __init__(self, partition, curr_depth):
        self.partition = partition
        self.features = self.partition.F
        self.curr_depth = curr_depth
        self.child = {}

        counts = [0, 0]
        for i, c in enumerate([-1, 1]):
            for e in self.partition.data:
                if e.label == c:
                    counts[i] += 1
        self.counts = counts

        if (self.partition.n == 0 or
            len(self.features) == 0 or
            self.curr_depth == DecisionTree.max_depth or
            counts[0] == 0 or counts[1] == 0):
            self.name = -1 if counts[0] >= counts[1] else 1
        else:
            s = self.best_feature()
            self.name = s
            for val in self.features[s]:
                # create a copy of features, then remove s
                d_features = copy.deepcopy(self.features)
                d_features.pop(s)
                # get a list of all examples that have s = val
                d = [e for e in self.partition.data if e.features[s] == val]
                d_partition = Partition(d, d_features)
                self.child[val] = DecisionTree(d_partition, self.curr_depth + 1)

    def best_feature(self):
        max_gain = 0.0
        best = None
        for x in self.features:
            if self.partition.info_gain(x) > max_gain:
                max_gain = self.partition.info_gain(x)
                best = x
        return best

    def predict(self, example):
        if len(self.child) == 0:
            return self.name

        key = self.name
        tokens = key.split('<=')
        val = None
        if len(tokens) > 1:
            name = tokens[0]
            thres = float(tokens[1])
            val = 'True' if example.features[name] <= thres else 'False'
        else:
            val = example.features[key]
        return self.child[val].predict(example)

    def __repr__(self):
        if len(self.child) == 0:
            return f'{self.counts}: {self.name}\n'

        rep = f'{self.counts}\n'
        for val in sorted(self.child.keys()):
            for i in range(self.curr_depth):
                rep += '|\t'
            rep += f'{self.name}={val} {self.child[val]}'
        return rep
