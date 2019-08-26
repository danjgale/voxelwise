"""Module for the Run class used by GLM classes"""

from nistats.design_matrix import make_first_level_design_matrix

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