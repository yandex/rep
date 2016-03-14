"""
These classes are wrappers for `XGBoost library <https://github.com/dmlc/xgboost>`_.
"""
from __future__ import division, print_function, absolute_import

import tempfile
import os
from abc import ABCMeta

import pandas
import numpy
from sklearn.utils import check_random_state

from .utils import normalize_weights, remove_first_line
from .interface import Classifier, Regressor
from .utils import check_inputs

__author__ = 'Mikhail Hushchyn, Alex Rogozhnikov'
__all__ = ['XGBoostBase', 'XGBoostClassifier', 'XGBoostRegressor']

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("please install xgboost")


class XGBoostBase(object):
    """
    A base class for the XGBoostClassifier and XGBoostRegressor. XGBoost tree booster is used.

    :param int n_estimators: number of trees built.
    :param int nthreads: number of parallel threads used to run XGBoost.
    :param num_feature: feature dimension used in boosting, set to maximum dimension of the feature
        (set automatically by XGBoost, no need to be set by user).
    :type num_feature: None or int
    :param float gamma: minimum loss reduction required to make a further partition on a leaf node of the tree.
        The larger, the more conservative the algorithm will be.
    :type gamma: None or float
    :param float eta: (or learning rate) step size shrinkage used in update to prevent overfitting.
        After each boosting step, we can directly get the weights of new features
        and eta actually shrinkages the feature weights to make the boosting process more conservative.
    :param int max_depth: maximum depth of a tree.
    :param float scale_pos_weight: ration of weights of the class 1 to the weights of the class 0.
    :param float min_child_weight: minimum sum of instance weight (hessian) needed in a child.
        If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
        then the building process will give up further partitioning.

        .. note:: weights are normalized so that mean=1 before fitting. Roughly min_child_weight is equal to the number of events.
    :param float subsample: subsample ratio of the training instance.
        Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees
        and this will prevent overfitting.
    :param float colsample: subsample ratio of columns when constructing each tree.
    :param float base_score: the initial prediction score of all instances, global bias.
    :param random_state: state for a pseudo random generator
    :type random_state: None or int or RandomState
    :param boot verbose: if 1, will print messages during training
    :param float missing: the number considered by XGBoost as missing value.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 n_estimators=100,
                 nthreads=16,
                 num_feature=None,
                 gamma=None,
                 eta=0.3,
                 max_depth=6,
                 scale_pos_weight=1.,
                 min_child_weight=1.,
                 subsample=1.,
                 colsample=1.,
                 base_score=0.5,
                 verbose=0,
                 missing=-999.,
                 random_state=0):

        self.n_estimators = n_estimators
        self.missing = missing
        self.nthreads = nthreads
        self.num_feature = num_feature
        self.gamma = gamma
        self.eta = eta
        self.max_depth = max_depth
        self.scale_pos_weight = scale_pos_weight
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample = colsample
        self.objective = None
        self.base_score = base_score
        self.verbose = verbose
        self.random_state = random_state
        self._num_class = None
        self.xgboost_estimator = None

    def _make_dmatrix(self, X, y=None, sample_weight=None):
        """
        Create XGBoost data from initial data.

        :return: XGBoost DMatrix
        """
        feature_names = [str(i) for i in range(X.shape[1])]
        matrix = xgb.DMatrix(data=X, label=y, weight=sample_weight,
                             missing=self.missing, feature_names=feature_names)
        return matrix

    def _check_fitted(self):
        assert self.xgboost_estimator is not None, "Classifier wasn't fitted, please call `fit` first"

    def _fit(self, X, y, estimator_type, sample_weight=None, **kwargs):
        """
        Train a classification/regression model on the data.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param y: labels of samples, array-like of shape [n_samples]
        :param sample_weight: weight of samples,
               array-like of shape [n_samples] or None if all weights are equal
        :param str estimator_type: type of the estimator (binary, reg or mult)
        :param dict kwargs: additional parameters
        :return: self
        """
        if self.random_state is None:
            seed = 0
        elif isinstance(self.random_state, int):
            seed = self.random_state
        else:
            seed = check_random_state(self.random_state).randint(0, 10000)

        self.objective = estimator_type
        params = {"nthread": self.nthreads,
                  "eta": self.eta,
                  "max_depth": self.max_depth,
                  "scale_pos_weight": self.scale_pos_weight,
                  "min_child_weight": self.min_child_weight,
                  "subsample": self.subsample,
                  "colsample_bytree": self.colsample,
                  "objective": self.objective,
                  "base_score": self.base_score,
                  "silent": int(not self.verbose),
                  "seed": seed}
        for key, value in kwargs.items():
            params[key] = value
            if key == 'num_class':
                self._num_class = value

        if self.num_feature is not None:
            params["num_feature"] = self.num_feature
        if self.gamma is not None:
            params["gamma"] = self.gamma

        xgboost_matrix = self._make_dmatrix(X, y, sample_weight)
        self.xgboost_estimator = xgb.train(params, xgboost_matrix, num_boost_round=self.n_estimators)

        return self

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['xgboost_estimator']
        if self.xgboost_estimator is None:
            result['dumped_xgboost'] = None
        else:
            with tempfile.NamedTemporaryFile() as dump:
                self._save_model(dump.name)
                with open(dump.name, 'rb') as dumpfile:
                    result['dumped_xgboost'] = dumpfile.read()
        return result

    def __setstate__(self, dict):
        self.__dict__ = dict
        if dict['dumped_xgboost'] is None:
            self.xgboost_estimator = None
        else:
            with tempfile.NamedTemporaryFile() as dump:
                with open(dump.name, 'wb') as dumpfile:
                    dumpfile.write(dict['dumped_xgboost'])
                self._load_model(dump.name)
            # HACK error in xgboost reloading
            if '_num_class' in dict:
                self.xgboost_estimator.set_param({'num_class': dict['_num_class']})
        del dict['dumped_xgboost']

    def _save_model(self, path_to_dump):
        """ Save XGBoost model"""
        self._check_fitted()
        self.xgboost_estimator.save_model(path_to_dump)

    def _load_model(self, path_to_dumped_model):
        """ Load XGBoost model to estimator """
        assert os.path.exists(path_to_dumped_model), 'there is no such file: {}'.format(path_to_dumped_model)
        self.xgboost_estimator = xgb.Booster({'nthread': self.nthreads}, model_file=path_to_dumped_model)

    def get_feature_importances(self):
        """
        Get features importances.

        :rtype: pandas.DataFrame with `index=self.features`
        """
        self._check_fitted()
        feature_score = self.xgboost_estimator.get_fscore()
        reordered_scores = numpy.zeros(len(self.features))
        for name, score in feature_score.items():
            reordered_scores[int(name)] = score
        return pandas.DataFrame({'effect': reordered_scores}, index=self.features)

    @property
    def feature_importances_(self):
        """Sklearn-way of returning feature importance.
        This returned as numpy.array, assuming that initially passed train_features=None """
        self._check_fitted()
        return self.get_feature_importances().ix[self.features, 'effect'].values


class XGBoostClassifier(XGBoostBase, Classifier):
    __doc__ = 'Implements classification model from XGBoost library. \n'\
              + remove_first_line(XGBoostBase.__doc__)

    def __init__(self, features=None,
                 n_estimators=100,
                 nthreads=16,
                 num_feature=None,
                 gamma=None,
                 eta=0.3,
                 max_depth=6,
                 scale_pos_weight=1.,
                 min_child_weight=1.,
                 subsample=1.,
                 colsample=1.,
                 base_score=0.5,
                 verbose=0,
                 missing=-999.,
                 random_state=0):

        XGBoostBase.__init__(self,
                             n_estimators=n_estimators,
                             nthreads=nthreads,
                             num_feature=num_feature,
                             gamma=gamma,
                             eta=eta,
                             max_depth=max_depth,
                             scale_pos_weight=scale_pos_weight,
                             min_child_weight=min_child_weight,
                             subsample=subsample,
                             colsample=colsample,
                             base_score=base_score,
                             verbose=verbose,
                             missing=missing,
                             random_state=random_state)

        Classifier.__init__(self, features=features)

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)
        sample_weight = normalize_weights(y, sample_weight=sample_weight, per_class=False)
        X = self._get_features(X)
        self._set_classes(y)
        if self.n_classes_ >= 2:
            return self._fit(X, y, 'multi:softprob', sample_weight=sample_weight, num_class=self.n_classes_)

    fit.__doc__ = Classifier.fit.__doc__

    def predict_proba(self, X):
        self._check_fitted()
        X_dmat = self._make_dmatrix(self._get_features(X))
        prediction = self.xgboost_estimator.predict(X_dmat, ntree_limit=0)
        if self.n_classes_ >= 2:
            return prediction.reshape(X.shape[0], self.n_classes_)

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X, step=None):
        """
        Predict probabilities for data for each class label on each stage..

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param int step: step for returned iterations (None by default).
            XGBoost does not implement this functionality and we need to
            predict from the beginning each time.
            With `None` passed step is chosen to have 10 points in the learning curve.
        :return: iterator

        .. warning: this method may be very slow, it takes iterations^2 / step time.
        """
        self._check_fitted()
        X_dmat = self._make_dmatrix(self._get_features(X))
        if step is None:
            step = max(self.n_estimators // 10, 1)

        # TODO use applying tree-by-tree
        for i in range(1, self.n_estimators // step + 1):
            prediction = self.xgboost_estimator.predict(X_dmat, ntree_limit=i * step)
            yield prediction.reshape(X.shape[0], self.n_classes_)


class XGBoostRegressor(XGBoostBase, Regressor):
    __doc__ = 'Implements regression model from XGBoost library. \n' + remove_first_line(XGBoostBase.__doc__)

    def __init__(self, features=None,
                 n_estimators=100,
                 nthreads=16,
                 num_feature=None,
                 gamma=None,
                 eta=0.3,
                 max_depth=6,
                 min_child_weight=1.,
                 subsample=1.,
                 colsample=1.,
                 objective_type='linear',
                 base_score=0.5,
                 verbose=0,
                 missing=-999.,
                 random_state=0):
        XGBoostBase.__init__(self,
                             n_estimators=n_estimators,
                             nthreads=nthreads,
                             num_feature=num_feature,
                             gamma=gamma,
                             eta=eta,
                             max_depth=max_depth,
                             min_child_weight=min_child_weight,
                             subsample=subsample,
                             colsample=colsample,
                             base_score=base_score,
                             verbose=verbose,
                             missing=missing,
                             random_state=random_state)

        Regressor.__init__(self, features=features)
        self.objective_type = objective_type

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)
        sample_weight = normalize_weights(y, sample_weight=sample_weight, per_class=False)
        X = self._get_features(X)
        assert self.objective_type in {'linear', 'logistic'}, 'Objective parameter is not valid'
        return self._fit(X, y, "reg:{}".format(self.objective_type), sample_weight=sample_weight)

    fit.__doc__ = Regressor.fit.__doc__

    def predict(self, X):
        self._check_fitted()
        X_dmat = self._make_dmatrix(self._get_features(X))
        return self.xgboost_estimator.predict(X_dmat, ntree_limit=0)

    predict.__doc__ = Regressor.predict.__doc__

    def staged_predict(self, X, step=None):
        """
        Predicts values for data on each stage.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param int step: step for returned iterations (None by default).
            XGBoost does not implement this functionality and we need to
            predict from the beginning each time.
            With `None` passed step is chosen to have 10 points in the learning curve.
        :return: iterator

        .. warning: this method may be very slow, it takes iterations^2 / step time.
        """
        self._check_fitted()
        X_dmat = self._make_dmatrix(self._get_features(X))
        if step is None:
            step = max(self.n_estimators // 10, 1)

        # TODO use applying tree-by-tree
        for i in range(1, self.n_estimators // step + 1):
            yield self.xgboost_estimator.predict(X_dmat, ntree_limit=i * step)
