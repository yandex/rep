from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

from nolearn.dbn import DBN
from sklearn.preprocessing import Imputer, MinMaxScaler
import numpy as np

__author__ = 'Alexey Berdnikov'


class NolearnClassifier(Classifier):
    """
    A wrapper for DBN from nolearn.dbn.

    Parameters:
    -----------
    :param features: features used in training
    :type features: `list[str]` or None
    :param layers: A list of ints of the form `[n_hid_units_1, n_hid_units_2, ...]`, where `n_hid_units_i` is the number
        of units in i-th hidden layer. The number of units in the input layer and the output layer will be set
        automatically. Default value is `[10]` which means one hidden layer containing 10 units.
    :param scales: Scale of the randomly initialized weights. A list of floating point values. When you find good values
        for the scale of the weights you can speed up training a lot, and also improve performance. Defaults to `0.05`.
    :param fan_outs: Number of nonzero incoming connections to a hidden unit. Defaults to `None`, which means that all
        connections have non-zero weights.
    :param output_act_funct: Output activation function. Instance of type :class:`gdbn.activationFunctions.Sigmoid`,
        :class:`gdbn.activationFunctions.Linear`, :class:`gdbn.activationFunctions.Softmax` from the
        :mod:`gdbn.activationFunctions` module.  Defaults to :class:`gdbn.activationFunctions.Softmax`.

    :param real_valued_vis: Set `True` (the default) if visible units are real-valued.
    :param use_re_lu: Set `True` to use rectified linear units. Defaults to `False`.
    :param uniforms: Not documented at this time.
    :param learn_rates: A list of learning rates, one entry per weight layer.
    :param learn_rate_decays: The number with which the `learn_rate` is multiplied after each epoch of fine-tuning.
    :param learn_rate_minimums: The minimum `learn_rates`; after the learn rate reaches the minimum learn rate, the
        `learn_rate_decays` no longer has any effect.
    :param momentum: Momentum
    :param l2_costs: L2 costs per weight layer.
    :param dropouts: Dropouts per weight layer.
    :param nesterov: Not documented at this time.
    :param nest_compare: Not documented at this time.
    :param rms_lims: Not documented at this time.
    :param learn_rates_pretrain: A list of learning rates similar to `learn_rates_pretrain`, but used for pretraining.
        Defaults to value of `learn_rates` parameter.
    :param momentum_pretrain: Momentum for pre-training. Defaults to value of `momentum` parameter.
    :param l2_costs_pretrain: L2 costs per weight layer, for pre-training.  Defaults to the value of `l2_costs`
        parameter.
    :param nest_compare_pretrain: Not documented at this time.
    :param epochs: Number of epochs to train (with backprop).
    :param epochs_pretrain: Number of epochs to pre-train (with CDN).
    :param loss_funct: A function that calculates the loss. Used for displaying learning progress.
    :param minibatch_size: Size of a minibatch.
    :param minibatches_per_epoch: Number of minibatches per epoch. The default is to use as many as fit into our
        training set.
    :param pretrain_callback: An optional function that takes as arguments the :class:`nolearn.dbn.DBN` instance, the
        epoch and the layer index as its argument, and is called for each epoch of pretraining.
    :param fine_tune_callback: An optional function that takes as arguments the :class:`nolearn.dbn.DBN` instance and
        the epoch, and is called for each epoch of fine tuning.
    :param verbose: Debugging output.
    .. warning::
        nolearn doesn't support `staged_predict_proba()`, `feature_importances__` and sample weights.

    """
    def __init__(self, features=None,
                 layers=(10,),
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

        self.layers = list(layers)
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
            self.imputer = Imputer()
            self.scaler = MinMaxScaler()

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
        del clf_params["layers"]
        clf_params["layer_sizes"] = [-1] + self.layers + [-1]

        return DBN(**clf_params)

    def fit(self, X, y):
        """
        Train the classifier.

        :param pandas.DataFrame | numpy.ndarray X: data shape `[n_samples, n_features]`
        :param list | numpy.array y: values - array-like of shape `[n_samples]`
        :return: self
        .. warning::
            Sample weights are not supported for nolearn.

        """
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
        """
        Predict data

        :param pandas.DataFrame | numpy.ndarray X: data shape `[n_samples, n_features]`
        :return: predicted values of shape `n_samples`

        """
        self._check_is_fitted()
        X = self._transform_data(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Predict data

        :param pandas.DataFrame | numpy.ndarray X: data shape `[n_samples, n_features]`
        :return: numpy.array of shape `[n_samples, n_classes]` with probabilities

        """
        self._check_is_fitted()
        X = self._transform_data(X)
        return self.clf.predict_proba(X)

    def staged_predict_proba(self, X):
        """

        :param pandas.DataFrame | numpy.ndarray X: data shape `[n_samples, n_features]`
        :return: iterator
        .. warning::
            Doesn't support for nolearn (**AttributeError** will be thrown).

        """
        raise AttributeError("'staged_predict_proba' is not supported for nolearn")
