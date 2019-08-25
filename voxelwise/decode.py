"""Decoding tools"""

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler,
                                   minmax_scale)
from sklearn.model_selection import (cross_val_score, LeaveOneGroupOut,
                                     permutation_test_score)


class Decode(object):
    def __init__(self, model, X, y, run_labels=None, kfold=None, scaling=None,
                 scaling_direction='voxels'):

        self.model = model
        self.scaling = scaling
        self.scaling_direction = scaling_direction

        self.X = X
        self.y = y
        self.run_labels = run_labels
        self.kfold = None

        if scaling_direction == 'voxels':
            if scaling == 'standardize':
                scaler = StandardScaler()
            elif scaling == 'minmax':
                scaler = MinMaxScaler()

            self.pipeline = Pipeline([('transformer', scaler),
                                      ('estimator', self.model)])

        elif scaling_direction == 'pattern':
            # transpose so that scaling is done along patterns
            self.X = scaler.fit_transform(X.T).T
            self.pipeline = Pipeline([('estimator', self.model)])


    def evaluate(self):

        if self.run_labels is None:
            self.accuracies = cross_val_score(self.pipeline, X=self.X, y=self.y,
                                              cv=self.kfold)
        else:
            loro = LeaveOneGroupOut()
            self.accuracies = cross_val_score(self.pipeline, X=self.X, y=self.y,
                                              groups=self.run_labels, cv=loro)

        self.mean_accuracy = np.mean(self.accuracies)


    def evaluate_permutation(self, n_permutations=100):

        if self.run_labels is None:
            results = permutation_test_score(self.pipeline, self.X, self.y,
                                             groups=self.run_labels,
                                             cv=self.kfold,
                                             n_permutations=n_permutations)
        else:
            loro = LeaveOneGroupOut()
            results = permutation_test_score(self.pipeline, self.X, self.y,
                                             groups=self.run_labels, cv=loro,
                                             n_permutations=n_permutations)
        # score returned from permutation is the average
        mean_accuracy, self.permutation_scores, self.p_vals = results

        if not hasattr(self, 'mean_accuracy'):
            self.mean_accuracy = mean_accuracy

        if not hasattr(self, 'accuracies'):
            self.accuracies = None


class GroupDecode(object):
    def __init__(self):
        pass


    def evaluate(self):
        pass


    def evaluate_permutation(self):
        pass


    def test_significance(self):
        pass


    def get_results(self):
        pass


    def save_results(self):
        pass


    def plot(self):
        pass
