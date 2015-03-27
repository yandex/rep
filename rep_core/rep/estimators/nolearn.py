from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

from nolearn.dbn import DBN
from sklearn.preprocessing import Imputer, MinMaxScaler
import numpy as np

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
                 verbose=0):
        Classifier.__init__(self, features=features)
        self.clf = None
        self.classes_ = None
        self.n_classes_ = None

        self.imputer = Imputer()
        self.scaler = MinMaxScaler()

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

        self.verbose = verbose

    def _transform_data(self, X, fit=False):
        data = self._get_train_features(X).values

        if fit:
            self.imputer.fit(data)
            self.scaler.fit(data)

        return self.scaler.transform(self.imputer.transform(data))

    def _set_classes(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y

    def _build_clf(self):
        clf_params = self.get_params()
        del clf_params["features"]

        return DBN(**clf_params)

    def fit(self, X, y):
        X, y, sample_weight = check_inputs(X, y, sample_weight=None, allow_none_weights=True)

        X = self._transform_data(X, fit=True)
        y = self._set_classes(y)

        self.clf = self._build_clf()
        self.clf.fit(X, y)

        return self

    def _check_is_fitted(self):
        if self.clf is None:
            raise AttributeError("estimator not fitted, call 'fit' before making predictions")

    def predict(self, X):
        self._check_is_fitted()
        X = self._transform_data(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted()
        X = self._transform_data(X)
        return self.clf.predict_proba(X)

    def staged_predict_proba(self, X):
        self._check_is_fitted()
        raise AttributeError("'staged_predict_proba' is not supported for nolearn")
