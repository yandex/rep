"""
:class:`SklearnClassifier` and :class:`SklearnRegressor` are wrappers for algorithms from scikit-learn.

From user perspective, wrapped sklearn model behaves in the same way as non-wrapped,
but has one additional parameter *features* to choose necessary columns to use in training.

Typically, models from **REP** are used with `pandas.DataFrames`,
which makes it possible to name needed variables or give some variables specific role in the training.

If data has :class:`numpy.array` type then behaviour will be the same as in sklearn.
For complete list of the available algorithms, see `sklearn API <http://scikit-learn.org/stable/modules/classes.html>`_.
"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs

__author__ = 'Alex Rogozhnikov'

__all__ = ['SklearnClassifier', 'SklearnRegressor']


class SklearnBase(object):
    """
    SklearnBase is a base for wrappers over sklearn-like models.
    All attributes will be returned for base estimator.
    """

    __metaclass__ = ABCMeta

    def __init__(self, clf):
        """
        :param clf: estimator to be used in training.
        """
        self.clf = clf

    def _fit(self, X, y, sample_weight=None, **kwargs):
        """
        Train the {estimator}.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: target of training, array-like of shape [n_samples]
        :param sample_weight: weight of samples,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self

        .. note:: if sklearn {estimator} doesn't support `sample_weight`, then put `sample_weight=None`,
            otherwise exception will be thrown.

        """
        if sample_weight is None:
            self.clf.fit(self._get_features(X), y, **kwargs)
        else:
            self.clf.fit(self._get_features(X), y, sample_weight=sample_weight, **kwargs)
        return self

    def __getattr__(self, name):
        # In order not to break pickling/unpickling.
        if name in ['__getstate__', '__setstate__']:
            raise AttributeError()
        # Those methods not defined here will be used directly from the estimator
        base = self.clf
        return base.__getattribute__(name)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Parameters of the base estimator can be accessed (for example param `depth`) by both *depth* and *clf__depth*.

        :param dict params: parameters to set in the model
        """
        params_for_clf = {}
        for name, value in params.items():
            if name == 'features':
                self.features = value
            elif name == 'clf':
                self.clf = value
            elif name.startswith('clf__'):
                params_for_clf[name[5:]] = value
            else:
                params_for_clf[name] = value
        self.clf.set_params(**params_for_clf)


class SklearnClassifier(SklearnBase, Classifier):
    def __init__(self, clf, features=None):
        """
        SklearnClassifier is a wrapper over sklearn-like **classifiers**.

        :param sklearn.BaseEstimator clf: classifier to train. Should be sklearn-compatible.
        :param features: features used in training
        :type features: list[str] or None
        """
        if isinstance(clf, Classifier):
            raise ValueError('Base class should be simply sklearn classifier, not descendant of Classifier')
        SklearnBase.__init__(self, clf=clf)
        Classifier.__init__(self, features=features)

    def fit(self, X, y, sample_weight=None, **kwargs):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        self._set_classes(y)
        return self._fit(X, y, sample_weight=sample_weight, **kwargs)

    fit.__doc__ = SklearnBase._fit.__doc__.format(estimator='classifier')

    def predict(self, X):
        return self.clf.predict(self._get_features(X))

    predict.__doc__ = Classifier.predict.__doc__

    def predict_proba(self, X):
        return self.clf.predict_proba(self._get_features(X))

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X):
        return self.clf.staged_predict_proba(self._get_features(X))

    staged_predict_proba.__doc__ = Classifier.staged_predict_proba.__doc__


class SklearnRegressor(SklearnBase, Regressor):
    def __init__(self, clf, features=None):
        """
        SklearnRegressor is a wrapper over sklearn-like **regressors**

        :param sklearn.BaseEstimator clf: classifier to train. Should be sklearn-compatible.
        :param features: features used in training
        :type features: list[str] or None
        """
        if isinstance(clf, Regressor):
            raise ValueError('Base class should be simply sklearn classifier, not descendant of Regressor')
        SklearnBase.__init__(self, clf=clf)
        Regressor.__init__(self, features=features)

    def fit(self, X, y, sample_weight=None, **kwargs):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        return self._fit(X, y, sample_weight=sample_weight, **kwargs)

    fit.__doc__ = SklearnBase._fit.__doc__.format(estimator='classifier')

    def predict(self, X):
        return self.clf.predict(self._get_features(X))

    predict.__doc__ = Regressor.predict.__doc__

    def staged_predict(self, X):
        """
        Predicts regression target at each stage for X.
        This method allows monitoring of error after each stage.

        :return: iterator
        """
        return self.clf.staged_predict(self._get_features(X))

    staged_predict.__doc__ = Regressor.staged_predict.__doc__
