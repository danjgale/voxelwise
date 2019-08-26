"""Classes for voxel pattern extraction via GLM approaches"""

import multiprocessing
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn._utils import check_niimg
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel

from voxelwise.run import Run


def _fit_glm(run):

    if run.separate_trial_index is not None:
        print('Fitting event {} for {}'.format(run.separate_trial_index,
                                               run.img.get_fname()))
    else:
        print('Fitting {}'.format(run.img.get_fname()))

    glm = FirstLevelModel(mask=run.mask)
    glm.fit(run.img, design_matrices=run.design)
    return glm


def _extract_parameters():
    # contrast = np.zeros(run.design.shape[1])
    # contrast[0] = 1
    # trial.beta_map = glm.compute_contrast(contrast, output_type='effect_size')
    # return trial
    pass


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

        return self


    def transform(self):

        if not self.__fit_status:
            print('LSS not yet fit. Please run .fit() first')

        parameter_imgs = _extract_parameters()
        img = nib.concat_images([i.beta_map for i in parameter_imgs])
        return img


    def fit_transform(self):
        self.fit()
        return self.transform()


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


class LSA(BaseGLM):
    def __init__(self):
        super().__init__(self)