"""Classes for voxel pattern extraction via GLM approaches"""

import multiprocessing
import numpy as np
import pandas as pd
from nilearn._utils import check_niimg
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel


def _fit_glm(run):

    if run.separate_trial_index is not None:
        print('Fitting event {} for {}'.format(run.separate_trial_index,
                                               run.img.get_fname()))
    else:
        print('Fitting {}'.format(run.img.get_fname()))

    glm = FirstLevelModel(mask=run.mask)
    glm.fit(run.img, design_matrices=run.design)
    return glm
    # contrast = np.zeros(run.design.shape[1])
    # contrast[0] = 1
    # trial.beta_map = glm.compute_contrast(contrast, output_type='effect_size')
    # return trial








class BaseGLM(object):
    def __init__(self, imgs, events, name, regressors=None, mask=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=1):

        self.imgs = check_niimg(imgs)
        self.events = events
        self.name = name
        self.regressors = regressors
        self.mask = mask
        self.trials = None
        self.t_r = t_r
        self.n_jobs = n_jobs
        self.__fit_status = False

        # to be set in child classes
        self.design = []
        self.imgs_for_fit = None


    def fit(self):

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        print('Fitting {} GLMs across {} cpus'.format(len(self.design), self.n_jobs))
        pool = multiprocessing.Pool(processes=self.n_jobs)
        try:
            self.models = pool.map(_fit_glm, self.designs)
            pool.close()
            self.__fit_status = True
        except Exception as e:
            print(e)
            pool.close()


    def transform(self):
        pass


    def fit_transform(self):
        pass


class Run(object):
    def __init__(self, img, events, regressors=None, mask=None,
                 separate_trial_index=None):
        self.img = img
        self.regressors = regressors

        if not isinstance(mask, list):
            mask = [mask] * len(img)
        self.mask = mask

        self.separate_trial_index = separate_trial_index
        if self.separate_trial_index is not None:
            ev = events.copy()
            ev.loc[separate_trial_index, 'trial_type'] = 'other'
            ev.loc[separate_trial_index, 'trial_type'] = 'interest'
            self.events = ev
        else:
            self.events = events


    def add_design_matrix(self, frame_times, hrf_model, drift_model=None,
                          period_cut=128):
        self.design = make_first_level_design_matrix(
            frame_times=frame_times,
            events=self.events,
            hrf_model=hrf_model,
            drift_model=drift_model,
            period_cut=period_cut
        )


def _lss_generator(img, event, regressors, mask=None, t_r=2,
                   hrf_model='spm + derivative', drift_model=None):

    n_scans = img.shape[3]
    frame_times = np.arange(n_scans) * t_r

    for ix in event.index.values():
        run = Run(img, event, regressors, mask, ix)
        run.add_design_matrix(frame_times, hrf_model, t_r, drift_model)
        yield run



class LSS(BaseGLM):
    def __init__(self, imgs, events, name, regressors=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=-1):
        super().__init__(self, imgs, events, name, regressors=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=-1)

        # efficiently map design matrices to their image
        self.designs = []
        for img, event, reg in zip(self.imgs, self.events, self.regressors):

            self.designs += _lss_generator(img, event, reg, t_r=self.t_r,
                                           hrf_model=self.hrf_model,
                                           drift_model=self.drift_model)




class LSS(object):
    def __init__(self, img, events, name, regressors=None,
                 hrf_model='spm + derivative', drift_model=None, t_r=2,
                 n_jobs=-1):
        """Generate Trial objects for each trial according to `self.events`.
        Each Trial object will have an associated design matrix for model
        fitting.
        """
        self.img = nib.load(img)
        self.events = events
        self.name = name
        self.trials = None
        self.t_r = t_r
        self.n_jobs = n_jobs
        self.__fit_status = False

        ### Eventually refactor this into fit_glm ---
        n_scans = self.img.get_data().shape[3]
        frame_times = np.arange(0, n_scans * 2, 2)
        self.trials = []
        for ix, row in self.events.iterrows():

            trial_params = row.to_dict()
            trial_params['number'] = ix + 1
            trial = Trial(**trial_params)

            ev = self.events.copy()
            ev.loc[ev['onset'] != row['onset'], 'trial_type'] = 'other'
            ev.loc[ev['onset'] == row['onset'], 'trial_type'] = 'current'

            design_kwargs = dict(frame_times=frame_times, events=ev,
                                 hrf_model=hrf_model, drift_model=drift_model,
                                 add_regs=regressors)
            design = make_first_level_design_matrix(**design_kwargs)
            trial.design = design
            self.trials.append(trial)
        ### ---


    @staticmethod
    def fit_glm(args):
        mask, img, trial, verbose = args
        glm = FirstLevelModel(mask=mask)
        if verbose:
            print('Fitting trial {}: onset-{}; {}'.format(trial.number,
                                                          trial.onset,
                                                          trial.trial_type))
        glm.fit(img, design_matrices=trial.design)
        contrast = np.zeros(trial.design.shape[1])
        contrast[0] = 1
        trial.beta_map = glm.compute_contrast(contrast,
                                              output_type='effect_size')
        return trial


    def fit(self, mask=None, verbose=False):

        self.mask = mask
        param_list = list(itertools.product([self.mask], [self.img], self.trials,
                                            [verbose]))


        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        print('Fitting {} GLMs across {} cpus'.format(len(param_list), self.n_jobs))
        pool = multiprocessing.Pool(processes=self.n_jobs)
        try:
            self.trials = pool.map(self.fit_glm, param_list)
            pool.close()
            self.__fit_status = True
        except Exception as e:
            print(e)
            pool.close()


    def transform(self):

        if not self.__fit_status:
            print('LSS not yet fit. Please run .fit() first')

        img = nib.concat_images([i.beta_map for i in self.trials])
        numbers = [i.number for i in self.trials]
        self.events['trial_number'] = numbers
        return self.events, img


    def fit_transform(self, mask=None, verbose=False):

        self.fit(mask=mask, verbose=verbose)
        return self.transform()


class LSA(BaseGLM):
    def __init__(self):
        super().__init__(self)