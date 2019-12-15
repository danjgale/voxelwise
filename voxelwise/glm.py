"""Classes for voxel pattern extraction via GLM approaches"""

import multiprocessing
from joblib import Parallel, delayed
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import math_img, concat_imgs
from nilearn._utils import check_niimg

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel

def _check_file(x):
    return pd.read_csv(x, sep='\t') if isinstance(x, str) else x


def _compute_frame_times(img, t_r):
    img = check_niimg(img)
    n_scans = img.shape[3]
    return np.arange(n_scans) * t_r


def _check_list(x, duplicate=None):
    x = x if isinstance(x, list) else [x]

    if duplicate is not None:
        return x * duplicate
    else:
        return x


class Model(object):
    def __init__(self, img, events, t_r, regressors=None, mask=None, 
                 standardize=False, signal_scaling=0, event_index=None,
                 first_level_kws=None):
        """Class for building a first-level model for the purpose of single-
        trial modelling.

        This is intended for single-trial modelling and not as a general
        purpose GLM class. To prevent unwanted aggregation, it only accepts one
        image/regressor/event/mask, rather than multiple. This is enforced and
        will raise a ValueError if inputs are a list of length > 1.

        Parameters
        ----------
        img : niimg-like
            One 4D functional image.
        events : pandas.core.DataFrame
            DataFrame with 'events', 'onsets' and 'trial_type' columns. Other
            columns can be included as parametric modulators. 
        regressors : pandas.core.DataFrame
            DataFrame with shape (number of volumes/timepoints in img, 
            number of regressors). Each column is a separate regressor. 
        mask : niimg-like
            Mask to restrict analysis, by default None which will compute
            a whole-brain mask. 
        event_index : int, optional
            Index of single event in `events`. All other events will be 
            labelled as 'other'. This is to isolate a single event for LSS
            modelling.  
        """

        self.img = img          
        self.mask = mask        
        
        regressors = _check_file(regressors)          
        
        if regressors is not None:
            self.regressors = regressors.values
            self.reg_names = regressors.columns.values
        else:
            self.regressors = regressors
            self.reg_names = None

        self.events = _check_file(events)

        # for LSS modelling
        self.event_index = event_index
        if self.event_index is not None:
            ev = events.copy()
            idx = ev.index.isin([event_index])
            ev.loc[~idx, 'trial_type'] = 'other'
            # to be used to index contrast during parameter extraction
            self._event_of_interest = ev.loc[idx, 'trial_type'].values[0]
            self.events = ev

        self.t_r = t_r
        self.frame_times = _compute_frame_times(self.img, self.t_r)

        if first_level_kws is not None:
            self.glm = FirstLevelModel(t_r=self.t_r, mask_img=mask, 
                                       standardize=standardize, 
                                       signal_scaling=signal_scaling, 
                                       **first_level_kws)
        else:
            self.glm = FirstLevelModel(t_r=self.t_r, mask_img=mask, 
                                       standardize=standardize, 
                                       signal_scaling=signal_scaling)
        self.design = None


    def add_design_matrix(self, hrf_model, drift_model='cosine', high_pass=.01):
        self.design = make_first_level_design_matrix(
            frame_times=self.frame_times,
            events=self.events,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass, 
            add_regs=self.regressors
        )
        return self


    def fit(self):
        self.glm.fit(self.img, design_matrices=self.design)
        return self


    def extract_params(self, param_type, contrast_ix=0):

        design = self.glm.design_matrices_[0]
        contrast = np.zeros(design.shape[1])
        contrast[contrast_ix] = 1

        if param_type == 'beta':
            param_img = self.glm.compute_contrast(contrast,
                                                  output_type='effect_size')
            reg_sd = np.std(design.iloc[:, contrast_ix])
            param_img = math_img("img * {}".format(reg_sd), img=param_img)
        
        elif param_type == 't':
            param_img = self.glm.compute_contrast(contrast, stat_type='t',
                                                  output_type='stat')
        else:
            param_img = self.glm.compute_contrast(contrast,
                                                  output_type=param_type)
        return param_img


def _fit_glm(model):
    if model.event_index is not None:
        print('Fitting event {} for {}'.format(model.event_index, model.img))
    else:
        print('Fitting {}'.format(model.img))
    model.fit()
    return model


class BaseGLM(object):
    def __init__(self, imgs, events, regressors=None, mask=None,
                 standardize=False, signal_scaling=0, 
                 hrf_model='spm + derivative', high_pass=.01, drift_model='cosine', 
                 t_r=2, n_jobs=1, first_level_kws=None):

        self.imgs = _check_list(imgs)

        # ensure appropriate types for imgs
        if not all(isinstance(item, str) for item in self.imgs):
            if all(isinstance(item, nib.spatialimages.SpatialImage) for item in self.imgs):
                warnings.warn('`imgs` is a list nibabel images, which will '
                              'use a much larger amount of memory than '
                              'providing a list of strings. For LSS this will '
                              'raise an error.', Warning)
            else:
                raise ValueError('`imgs` must be a list of string or '
                                 'nibabel.spatialimages.SpatialImage (except '
                                 'LSS, which enforces only strings)')    

        self.events = _check_list(events)

        if regressors is None:
            self.regressors = _check_list(regressors, duplicate=len(imgs))
        else:
            self.regressors = _check_list(regressors)
        
        self.mask = mask
        self.standardize = standardize
        self.signal_scaling = signal_scaling
        self.hrf_model = hrf_model
        self.high_pass = high_pass
        self.drift_model = drift_model
        self.t_r = t_r
        self.n_jobs = n_jobs
        self.first_level_kws = first_level_kws
        
        self._fit_status = False

        # to be set in child classes
        self.models = []
        self.imgs_for_fit = None


    def fit(self):

        if self.n_jobs == 1:
            print('Fitting {} GLMs serially'.format(len(self.models)))
            self.models = [self.fit_glm(model) for model in self.models]
            self._fit_status = True
        else:
            if self.n_jobs == -1:
                self.n_jobs = multiprocessing.cpu_count()
            
            print('Fitting {} GLMs across {} cpus'.format(len(self.models),
                                                              self.n_jobs))

            self.models = Parallel(self.n_jobs)(delayed(_fit_glm)(x) 
                                                for x in self.models)

        return self


    def transform(self, param_type='z_score'):
        if not self._fit_status:
            print('BaseGLM not yet fit. Please run .fit() first')
        return self


    def fit_transform(self, param_type='z_score'):
        self.fit()
        return self.transform(param_type)


def _lss_generator(img, event, regressors, mask=None, standardize=False, 
                   signal_scaling=0, t_r=2, high_pass=.01,
                   hrf_model='spm + derivative', drift_model='cosine', 
                   first_level_kws=None):
    """Generate a new first level model for each event of a single image"""

    if isinstance(event, str):
        event = pd.read_csv(event, sep='\t')

    # iterate through each event and create model
    for ix in event.index.values:
        model = Model(img, event, t_r, regressors, mask, standardize, 
                      signal_scaling, ix, first_level_kws)
        model.add_design_matrix(hrf_model, drift_model, high_pass)
        yield model


class LSS(BaseGLM):
    def __init__(self, imgs, events, regressors=None, mask=None, t_r=2,
                 standardize=False, signal_scaling=0, 
                 hrf_model='spm + derivative', high_pass=.01, drift_model='cosine', 
                 n_jobs=-1, first_level_kws=None):
        super().__init__(imgs, events, regressors, mask, standardize, 
                         signal_scaling, hrf_model, high_pass, drift_model, 
                         t_r, n_jobs, first_level_kws)

        if not all(isinstance(item, str) for item in self.imgs):
            raise ValueError('LSS forces `imgs` to be a list of strings to '
                             'reduce memory load of generating a large number '
                             'of GLMs. Nibabel objects are not permitted.')

        # one model per trial (many models per image)
        self.models = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            self.models += _lss_generator(img, event, reg, mask=self.mask, 
                                          standardize=self.standardize, 
                                          signal_scaling=self.signal_scaling, 
                                          t_r=self.t_r, high_pass=self.high_pass, 
                                          hrf_model=self.hrf_model,
                                          drift_model=self.drift_model, 
                                          first_level_kws=self.first_level_kws)


    def transform(self, param_type='z_score'):
        if not self._fit_status:
            raise NotImplementedError('LSS not yet fit. Please run .fit() first')

        param_maps = []
        list_ = []
        for model in self.models:

            if not isinstance(model.img, str):
                raise Exception('{} not a string'.format(model.img))

            # ensure that we get event of LSS using a lookup
            ev = model.design.columns.get_loc(model._event_of_interest)
            param_maps.append(model.extract_params(param_type, contrast_ix=ev))
            # get trial info for the map
            event = model.events.loc[model.event_index]
            list_.append({'src_img': model.img,
                          'trial_type': event['trial_type'],
                          'onset': event['onset']})
        param_index = pd.DataFrame(list_)

        return concat_imgs(param_maps), param_index


def _rename_lsa_trial_types(df):
    """Combine event name and onset for a unique event label"""

    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')

    df = df.copy()
    unique_labels = df['trial_type'] + '___' + df['onset'].round(2).astype(str)
    df['trial_type'] = unique_labels
    return df


class LSA(BaseGLM):
    def __init__(self, imgs, events, regressors=None, mask=None, t_r=2,
                 standardize=False, signal_scaling=0,
                 hrf_model='spm + derivative', high_pass=.01, drift_model='cosine', 
                 n_jobs=-1, first_level_kws=None):
        super().__init__(imgs, events, regressors, mask, standardize, 
                         signal_scaling, hrf_model, high_pass, drift_model, 
                         t_r, n_jobs, first_level_kws)

        # one model per image
        self.models = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            event = _rename_lsa_trial_types(event)

            model = Model(img, event, self.t_r, reg, self.mask, 
                          self.standardize, self.signal_scaling, 
                          first_level_kws=self.first_level_kws)
            model.add_design_matrix(self.hrf_model, self.drift_model, 
                                    self.high_pass)
            self.models.append(model)


    def transform(self, param_type='z_score'):
        if not self._fit_status:
            raise NotImplementedError('LSA not yet fit. Please run .fit() first')

        param_maps = []
        list_ = []
        for model in self.models:
            # iterate only through trial_types
            trial_reg_names = model.events['trial_type'].tolist()
            if len(np.unique(trial_reg_names)) != len(trial_reg_names):
                raise Exception('Trial regressor names are not unique')

            for ev in trial_reg_names:
                reg = model.design.columns.get_loc(ev)
                param_maps.append(model.extract_params(param_type,
                                                       contrast_ix=reg))
                trial_type, onset = model.design.columns[reg].split('___')
                list_.append({'src_img': model.img, 'trial_type': trial_type, 
                              'onset': onset})

        param_index = pd.DataFrame(list_)

        return concat_imgs(param_maps), param_index


class LSU(BaseGLM):
    def __init__(self, imgs, events, regressors=None, mask=None, t_r=2,
                 standardize=False, signal_scaling=0,
                 hrf_model='spm + derivative', high_pass=.01, drift_model='cosine', 
                 n_jobs=-1, first_level_kws=None):
        super().__init__(imgs, events, regressors, mask, standardize, 
                         signal_scaling, hrf_model, high_pass, drift_model, 
                         t_r, n_jobs, first_level_kws)

        # one model per imge
        self.models = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            # event trial_types are kept the same
            model = Model(img, event, self.t_r, reg, self.mask, 
                          self.standardize, self.signal_scaling, 
                          first_level_kws=self.first_level_kws)
            model.add_design_matrix(self.hrf_model, self.drift_model, 
                                    self.high_pass)
            self.models.append(model)


    def transform(self, param_type='z_score'):
        if not self._fit_status:
            print('LSU not yet fit. Please run .fit() first')

        param_maps = []
        list_ = []
        for model in self.models:

            # iterate only through unique trial_types
            trial_reg_names = np.unique(model.events['trial_type'])
            for ev in trial_reg_names:
                reg = model.design.columns.get_loc(ev)
                param_maps.append(model.extract_params(param_type,
                                                       contrast_ix=reg))
                list_.append({'src_img': model.img, 'trial_type': ev})

        param_index = pd.DataFrame(list_)

        return concat_imgs(param_maps), param_index