"""Decoding tools"""

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler,
                                   minmax_scale)
from sklearn.model_selection import (cross_val_score, LeaveOneGroupOut,
                                     permutation_test_score, LeaveOneOut, 
                                     KFold)


class Decode(object):
    def __init__(self, model, X, y, run_labels=None, scaling=None,
                 scaling_direction='voxels'):

        self.model = model
        self.scaling = scaling
        self.scaling_direction = scaling_direction

        self.X = X
        self.y = y
        self.run_labels = run_labels

        if scaling == 'standardize':
                scaler = StandardScaler()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
        elif scaling == 'center':
            scaler = StandardScaler(with_std=False)


        # scale within each pattern if specified
        if (scaling_direction == 'pattern') | (scaling_direction == 'both'):
            self.X = scaler.fit_transform(X.T).T
        
        # set model with or without feature scaling 
        if (scaling_direction == 'voxels') | (scaling_direction == 'both'):
            self.pipeline = Pipeline([('transformer', scaler),
                                      ('estimator', self.model)])
        else:
            self.pipeline = Pipeline([('estimator', self.model)]) 
        

    @staticmethod
    def get_cross_val_scheme(scheme):
        if isinstance(scheme, int):
            return KFold(scheme)
        elif scheme == 'run':
            return LeaveOneGroupOut()
        elif scheme == 'one':
            return LeaveOneOut()
        else:
            raise ValueError('Incorrect cross validation scheme specified')


    def evaluate(self, cross_val_scheme='run'):

        self.cross_validator = self.get_cross_val_scheme(cross_val_scheme)

        if cross_val_scheme == 'run':
            if self.run_labels is None:
                raise ValueError("run_labels must not be None if 'run' is"
                                 " selected for cross_val_scheme")
        else:
            # ensure that data is not grouped
            self.run_labels = None
        
        self.accuracies = cross_val_score(self.pipeline, X=self.X, y=self.y,
                                            groups=self.run_labels, 
                                            cv=self.cross_validator)

        return np.mean(self.accuracies)


    def evaluate_permutation(self, cross_val_scheme='run', n_permutations=100):

        if not hasattr(self, 'cross_validator'):
            self.cross_validator = self.get_cross_val_scheme(cross_val_scheme)

        if cross_val_scheme == 'run':
            if self.run_labels is None:
                raise ValueError("run_labels must not be None if 'run' is"
                                 " selected for cross_val_scheme")
        else:
            # ensure that data is not grouped
            self.run_labels = None

        results = permutation_test_score(self.pipeline, X=self.X, y=self.y,
                                            groups=self.run_labels, 
                                            cv=self.cross_validator,
                                            n_permutations=n_permutations)


        if not hasattr(self, 'accuracies'):
            self.accuracies = None

        # mean_accuracy, permutation_scores, p_vals
        return results


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
