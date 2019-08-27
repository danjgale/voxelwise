"""Classes for voxel pattern extraction via GLM approaches"""

import multiprocessing
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import math_img, concat_imgs
from nilearn._utils import check_niimg

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel


class Model(object):
    def __init__(self, img, events, regressors=None, mask=None,
                 event_index=None):
        """Class for building a first-level model for the purpose of single-
        trial modelling.

        This is intended for single-trial modelling and not as a general
        purpose GLM class. To prevent unwanted aggregation, it only accepts one
        image/regressor/event/mask, rather than multiple. This is enforced and
        will raise a ValueError if inputs are a list of length > 1.

        Parameters
        ----------
        img : [type]
            [description]
        events : [type]
            [description]
        regressors : [type], optional
            [description], by default None
        mask : [type], optional
            [description], by default None
        event_index : [type], optional
            [description], by default None
        """

        if isinstance(img, list) & len(img) > 1:
            raise ValueError('Model does not accept lists with length > 1')
        self.img = img

        if isinstance(regressors, list) & len(regressors) > 1:
            raise ValueError('Model does not accept lists with length > 1')
        self.regressors = regressors

        if isinstance(mask, list) & len(mask) > 1:
            raise ValueError('Model does not accept lists with length > 1')
        self.mask = mask

        self.event_index = event_index
        if self.event_index is not None:
            ev = events.copy()
            ev.loc[~ev.index.isin([event_index]), 'trial_type'] = 'other'
            self.events = ev
        else:
            self.events = events

        self.glm = FirstLevelModel(mask=self.mask)
        self.design = None


    def add_design_matrix(self, frame_times, hrf_model, drift_model=None,
                          period_cut=128):
        self.design = make_first_level_design_matrix(
            frame_times=frame_times,
            events=self.events,
            hrf_model=hrf_model,
            drift_model=drift_model,
            period_cut=period_cut
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
        else:
            param_img = self.glm.compute_contrast(contrast,
                                                  output_type=param_type)
        return param_img


class BaseGLM(object):
    def __init__(self, imgs, events, name, regressors=None, mask=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 period_cut=128, n_jobs=1):

        self.imgs = check_niimg(imgs)
        self.events = events
        self.name = name
        self.regressors = regressors
        self.mask = mask
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.t_r = t_r
        self.period_cut = period_cut
        self.n_jobs = n_jobs
        self._fit_status = False

        # to be set in child classes
        self.models = []
        self.imgs_for_fit = None


    @staticmethod
    def fit_glm(model):

        if model.event_index is not None:
            print('Fitting event {} for {}'.format(model.event_index,
                                                   model.img.get_filename()))
        else:
            print('Fitting {}'.format(model.img.get_fname()))

        model.fit()
        return model


    def fit(self):

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        print('Fitting {} GLMs across {} cpus'.format(len(self.models),
                                                      self.n_jobs))
        pool = multiprocessing.Pool(processes=self.n_jobs)
        try:
            self.models = pool.map(self.fit_glm, self.models)
            pool.close()
            self._fit_status = True
        except Exception as e:
            print(e)
            pool.close()

        return self


    def transform(self, param_type='beta'):
        if not self._fit_status:
            print('BaseGLM not yet fit. Please run .fit() first')
        return self


    def fit_transform(self):
        self.fit()
        return self.transform()


def _lss_generator(img, event, regressors, mask=None, t_r=2,
                   hrf_model='spm + derivative', drift_model=None):
    """Generate a new first level model for each event of a single image"""

    n_scans = img.shape[3]
    frame_times = np.arange(n_scans) * t_r

    for ix in event.index.values():
        model = Model(img, event, regressors, mask, ix)
        model.add_design_matrix(frame_times, hrf_model, t_r, drift_model)
        yield model


class LSS(BaseGLM):
    def __init__(self, imgs, events, name, regressors=None, mask=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=-1):
        super().__init__(self, imgs, events, name, regressors=None, mask=mask,
                         hrf_model='spm + derivative', drift_model=None, t_r=2,
                         n_jobs=-1)

        # one model per trial (many models per image)
        self.models = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            self.models += _lss_generator(img, event, reg, t_r=self.t_r,
                                          hrf_model=self.hrf_model,
                                          drift_model=self.drift_model)


    def transform(self, param_type='beta'):
        if not self._fit_status:
            print('LSS not yet fit. Please run .fit() first')

        param_maps = []
        list_ = []
        for model in self.models:
            param_maps.append(model.extract_params(param_type))
            # get trial info for the map
            event = model.events.loc[model.event_index]
            list_.append({'img_name': model.img.get_filename(),
                          'trial_type': event['trial_type'],
                          'onset': event['onset']})
        param_index = pd.DataFrame(list_)

        return concat_imgs(param_maps), param_index


def _rename_lsa_trial_types(df):
    """Combine event name and onset for a unique event label"""
    df = df.copy()
    unique_labels = df['trial_type'] + '_' + df['onset']
    df['trial_type'] = unique_labels
    return df


class LSA(BaseGLM):
    def __init__(self, imgs, events, name, regressors=None, mask=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=-1):
        super().__init__(self, imgs, events, name, regressors=None, mask=mask,
                         hrf_model='spm + derivative', drift_model=None, t_r=2,
                         n_jobs=-1)

        # one model per imge
        self.models = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            n_scans = img.shape[3]
            frame_times = np.arange(n_scans) * self.t_r

            event = _rename_lsa_trial_types(event)
            model = Model(img, event, reg, mask)
            model.add_design_matrix(frame_times, self.hrf_model,
                                    self.drift_model, self.period_cut)
            self.models.append(model)


    def transform(self, param_type='beta'):
        if not self._fit_status:
            print('LSA not yet fit. Please run .fit() first')

        param_maps = []
        list_ = []
        for model in self.models:
            # iterate only through trial_types
            trial_reg_names = model.events['trial_types'].tolist()
            if len(np.unique(trial_reg_names)) != len(trial_reg_names):
                raise Exception('Trial regressor names are not unique')

            for ev in trial_reg_names:
                reg = model.design.columns.index(ev)
                param_maps.append(model.extract_params(param_type,
                                                       contrast_ix=reg))
                trial_type, onset = model.design.columns[reg].split('_')
                list_.append({'img_name': model.img.get_filename(),
                              'trial_type': trial_type, 'onset': onset})

        param_index = pd.DataFrame(list_)

        return concat_imgs(param_maps), param_index


class LSU(BaseGLM):
    def __init__(self, imgs, events, name, regressors=None, mask=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=-1):
        super().__init__(self, imgs, events, name, regressors=None, mask=mask,
                         hrf_model='spm + derivative', drift_model=None, t_r=2,
                         n_jobs=-1)

        # one model per imge
        self.models = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            n_scans = img.shape[3]
            frame_times = np.arange(n_scans) * self.t_r

            # event trial_types are kept the same
            model = Model(img, event, reg, mask)
            model.add_design_matrix(frame_times, self.hrf_model,
                                    self.drift_model, self.period_cut)
            self.models.append(model)


    def transform(self, param_type='beta'):
        if not self._fit_status:
            print('LSU not yet fit. Please run .fit() first')

        param_maps = []
        list_ = []
        for model in self.models:
            # iterate only through unique trial_types
            trial_reg_names = np.unique(model.events['trial_types'])
            for ev in trial_reg_names:
                reg = model.design.columns.index(ev)
                param_maps.append(model.extract_params(param_type,
                                                       contrast_ix=reg))
                trial_type, onset = model.design.columns[reg].split('_')
                list_.append({'img_name': model.img.get_filename(),
                              'trial_type': trial_type, 'onset': onset})

        param_index = pd.DataFrame(list_)

        return concat_imgs(param_maps), param_index