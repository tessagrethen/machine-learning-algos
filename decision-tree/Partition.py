"""
Authors: Sara Mathieson, Tessa Pham
Description: Implements Partition and Example classes.
Date: 09/12/2019
"""

import math

class Example:

    def __init__(self, features, label):
        """Helper class that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data, F):
        """Store information about a dataset."""
        self.data = data # list of Examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)

    def prob_y(self, c):
        """Calculate the probability of a label out of all examples."""
        count = 0
        for e in self.data:
            if e.label == c:
                count += 1
        return count / self.n

    def prob_yx(self, c, x, v):
        """Calculate the probability of a label given a feature value."""
        count = 0
        for e in self.data:
            if e.label == c and e.features[x] == v:
                count += 1
        p = count / self.n
        px = self.prob_x(x, v)
        return 0 if px == 0 else p / px

    def prob_x(self, x, v):
        """Calculate the probability of a feature value out of all examples."""
        count = 0
        for e in self.data:
            if e.features[x] == v:
                count += 1
        return count / self.n

    def entropy(self):
        """Calculate the entropy of Y."""
        h = 0.0
        for c in [-1, 1]:
            p = self.prob_y(c)
            h += -(p * math.log(p, 2))
        return h

    def cond_entropy_x(self, x):
        """Calculate the entropy of Y given a feature X."""
        h = 0.0
        for v in self.F[x]:
            p = self.prob_x(x, v)
            h += p * self.cond_entropy_xv(x, v)
        return h

    def cond_entropy_xv(self, x, v):
        """Calculate the entropy of Y given a feature value X = v."""
        h = 0.0
        for c in [-1, 1]:
            p = self.prob_yx(c, x, v)
            if p:
                h += -(p * math.log(p, 2))
        return h

    def info_gain(self, x):
        """Calculate the information gain for a feature."""
        return self.entropy() - self.cond_entropy_x(x)
