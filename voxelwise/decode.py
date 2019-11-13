"""Decoding tools"""

import multiprocessing
from functools import partial
from itertools import repeat
import numpy as np
from scipy import stats
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler,
                                   minmax_scale)
from sklearn.model_selection import (cross_val_score, LeaveOneGroupOut,
                                     permutation_test_score, LeaveOneOut, 
                                     KFold)
import nibabel as nib                                     
from nilearn.input_data import NiftiMasker, MultiNiftiMasker

# from voxelwise.report import make_report


def _check_input_data(data, mask_img=None, return_first_element=False):
    if not isinstance(data, list):
        data = [data]
    
    if all(isinstance(x, nib.spatialimages.SpatialImage) for x in data):
        masker = MultiNiftiMasker(mask_img)
        data = masker.fit_transform(data)

    elif all(isinstance(x, np.ndarray) for x in data):
        pass

    else:
        raise ValueError('input_data must be an instance of numpy.ndarray or '
                         'nibabel.spatialimages.SpatialImage')
    
    # when being used for Decode -- the actual image/array is needed
    if return_first_element:
        return data[0]
    else:
        return data


def _get_cross_val_scheme(scheme):
    if isinstance(scheme, int):
        return KFold(scheme)
    elif scheme == 'run':
        return LeaveOneGroupOut()
    elif scheme == 'one':
        return LeaveOneOut()
    else:
        raise ValueError('Incorrect cross validation scheme specified')


class Decode(object):
    def __init__(self, model, mask_img=None, cross_val_scheme='run', 
                 scaling=None, scaling_direction='voxels', n_permutations=None):

        self.model = model
        self.mask_img = mask_img
        self.scaling = scaling
        self.scaling_direction = scaling_direction
        self.cross_val_scheme = cross_val_scheme
        self.n_permutations = n_permutations

        if scaling == 'standardize':
            self.scaler = StandardScaler()
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'center':
            self.scaler = StandardScaler(with_std=False)

        # set model with or without feature scaling 
        if (scaling_direction == 'voxels') | (scaling_direction == 'both'):
            self.pipeline = Pipeline([('transformer', self.scaler),
                                      ('estimator', self.model)])
        else:
            self.pipeline = Pipeline([('estimator', self.model)])

        self.__fit_status = False


    def fit(self, X, y, run_labels=None):
        
        self.X = _check_input_data(X, mask_img=self.mask_img, 
                                   return_first_element=True)
        self.y = y
        self.run_labels = run_labels
        
        # scale within each pattern if specified
        if (self.scaling_direction == 'pattern') | (self.scaling_direction == 'both'):
            self.X = self.scaler.fit_transform(X.T).T

        if self.cross_val_scheme == 'run':
            if self.run_labels is None:
                raise ValueError("run_labels must not be None if 'run' is"
                                 " selected for cross_val_scheme")
        else:
            # ensure that data is not grouped
            self.run_labels = None
        
        cross_validator = _get_cross_val_scheme(self.cross_val_scheme)
        if self.n_permutations is not None:
            res = permutation_test_score(self.pipeline, X=self.X, y=self.y,
                                         groups=self.run_labels, 
                                         cv=cross_validator, 
                                         n_permutations=self.n_permutations)
            self.accuracies_, self.permutation_scores_, self.pval_ = res
        else:
            self.accuracies_ = cross_val_score(self.pipeline, X=self.X, y=self.y,
                                               groups=self.run_labels, 
                                               cv=cross_validator)
            self.permutation_scores_ = None
            self.pval_ = None

        self.__fit_status = True


    def transform(self):
        if not self.__fit_status:
            raise NotImplementedError('Decode has not been fit yet')
        else:
            return self.accuracies_
        

    def fit_transform(self, X, y, run_labels=None):
        self.fit(X, y, run_labels)
        return self.transform()


class GroupDecode(object):
    def __init__(self, model, mask_img=None, cross_val_scheme='run', 
                 scaling=None, scaling_direction='voxels', n_permutations=None,
                 n_jobs=1):
        
        self.model = model
        self.mask_img = mask_img
        self.scaling = scaling
        self.scaling_direction = scaling_direction
        self.cross_val_scheme = cross_val_scheme
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs

        self.__fit_status = False


    @staticmethod
    def decode_subject(decoder, X, y, run_labels):
        decoder.fit(X, y, run_labels)
        return decoder.accuracies_, decoder.permutation_scores_, decoder.pval_


    def fit(self, X, y, run_labels=None):

        self.X = _check_input_data(X, self.mask_img)
        self.y = y
        self.run_labels = run_labels
        
        if self.run_labels is None:
            run_labels = [run_labels]
            run_labels *= len(self.X)
        else:
            run_labels = self.run_labels

        decoder = Decode(self.model, self.mask_img, self.cross_val_scheme, 
                         self.scaling, self.scaling_direction, 
                         self.n_permutations)
        if self.n_jobs == 1:

            self.accuracies_ = []
            self.permutation_scores_ = []
            self.pval_ = [] 
            for features, response, labels in zip(self.X, self.y, run_labels):
                res = self.decode_subject(decoder, features, response, labels)
                self.accuracies_.append(res[0])
                self.permutation_scores_.append(res[1])
                self.pval_.append(res[2])

            self.__fit_status = True

        else:   
            if self.n_jobs == -1:
                self.n_jobs = multiprocessing.cpu_count()

            pool = multiprocessing.Pool(processes=self.n_jobs)
            try:
                args = list(zip(repeat(decoder, len(self.X)), self.X, self.y, 
                            run_labels))
                results = pool.starmap(self.decode_subject,  args)
                pool.close()

                # unpack into separate lists
                results = [list(x) for x in list(zip(*results))]
                self.accuracies_, self.permutation_scores_, self.pval_ = results
                
                self.__fit_status = True
            
            except Exception as e:
                print(e)
                pool.close()


    def transform(self):
        if not self.__fit_status:
            raise NotImplementedError('Decode has not been fit yet')
        else:
            return self.accuracies_
        

    def fit_transform(self, X, y, run_labels=None):
        self.fit(X, y, run_labels)
        return self.transform()


    def make_results(self):
        if not self.__fit_status:
            raise NotImplementedError('Decode has not been fit yet')

        list_ = []
        for i, acc in enumerate(self.accuracies_):
            list_.append(pd.DataFrame({'subject': i, 'accuracy': acc, 
                                       'fold': np.arange(len(acc))}))
        return pd.concat(list_)


    # def get_report(self):
    #     df = self.make_results()
    #     make_report(df)


    def save(self, fname):
        df = self.make_results()
        df.to_table(fname)


    def plot(self):
        pass
