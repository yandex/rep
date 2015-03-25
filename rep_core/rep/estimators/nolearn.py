from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

from nolearn.dbn import DBN
from sklearn.preprocessing import Imputer, MinMaxScaler

import inspect

__author__ = 'Alexey Berdnikov'


class NolearnClassifier(Classifier):
    def __init__(self, features=None,
                 layer_sizes=None,
                 scales=0.05,
                 fan_outs=None,
                 output_act_funct=None,
                 real_valued_vis=True,
                 use_re_lu=True,
                 uniforms=False,
                 learn_rates=0.1,
                 learn_rate_decays=1.0,
                 learn_rate_minimums=0.0,
                 momentum=0.9,
                 l2_costs=0.0001,
                 dropouts=0,
                 nesterov=True,
                 nest_compare=True,
                 rms_lims=None,
                 learn_rates_pretrain=None,
                 momentum_pretrain=None,
                 l2_costs_pretrain=None,
                 nest_compare_pretrain=None,
                 epochs=10,
                 epochs_pretrain=0,
                 loss_funct=None,
                 minibatch_size=64,
                 minibatches_per_epoch=None,
                 pretrain_callback=None,
                 fine_tune_callback=None,
                 random_state=None,
                 verbose=0):
        Classifier.__init__(self, features=features)
        self.clf = None

        self.layer_sizes = layer_sizes
        self.scales = scales
        self.fan_outs = fan_outs
        self.output_act_funct = output_act_funct
        self.real_valued_vis = real_valued_vis
        self.use_re_lu = use_re_lu
        self.uniforms = uniforms

        self.learn_rates = learn_rates
        self.learn_rate_decays = learn_rate_decays
        self.learn_rate_minimums = learn_rate_minimums
        self.momentum = momentum
        self.l2_costs = l2_costs
        self.dropouts = dropouts
        self.nesterov = nesterov
        self.nest_compare = nest_compare
        self.rms_lims = rms_lims

        self.learn_rates_pretrain = learn_rates_pretrain
        self.momentum_pretrain = momentum_pretrain
        self.l2_costs_pretrain = l2_costs_pretrain
        self.nest_compare_pretrain = nest_compare_pretrain

        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.loss_funct = loss_funct
        self.minibatch_size = minibatch_size
        self.minibatches_per_epoch = minibatches_per_epoch

        self.pretrain_callback = pretrain_callback
        self.fine_tune_callback = fine_tune_callback

        self.random_state = random_state

        self.verbose = verbose

    def _get_param_names(self):
        return inspect.getargspec(self.__init__)[0][1:]

    def get_params(self):
        params = {}
        param_names = self._get_param_names()
        for key in param_names:
            value = getattr(self,key)
            params[key] = value
        return params

    def set_params(self, **params):
        valid_param_names = self._get_param_names()
        for key, value in params.items():
            if key not in valid_param_names:
                raise ValueError("NolearnClassifier has no parameter '{}'".format(key))
            else:
                setattr(self, key, value)

    def _transform_data(self, X, fit=False):
        data = self._get_train_features(X).values

        if fit:
            self.imputer = Imputer()
            self.scaler = MinMaxScaler()

            self.imputer.fit(data)
            self.scaler.fit(data)

        return self.scaler.transform(self.imputer.transform(data))

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        if sample_weight is not None:
            raise ValueError("'sample_weight' parameter is not supported for nolearn")
        X = self._transform_data(X, fit=True)

        clf_params = self.get_params()
        del clf_params["features"]
        self.clf = DBN(**clf_params)

        self.clf.fit(X, y)

        return self

    def _check_fitted(self):
        if self.clf is None:
            raise ValueError("estimator not fitted, call 'fit' before making predictions.")

    def predict(self, X):
        self._check_fitted()
        X = self._transform_data(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        self._check_fitted()
        X = self._transform_data(X)
        return self.clf.predict_proba(X)

    def staged_predict_proba(self, X):
        self._check_fitted()
        raise ValueError("'staged_predict_proba' is not supported for nolearn")
