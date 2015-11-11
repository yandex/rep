"""
Sklearn wrapper for users is the same as sklearn model,
has only one additional parameter *features* to choose necessary columns for training.
If data has :class:`numpy.array` type then behaviour will be the same as in sklearn.
For complete list of available algorithms, see `sklearn API <http://scikit-learn.org/stable/modules/classes.html>`_.
"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta

from .interface import Classifier, Regressor
from .utils import check_inputs


__author__ = 'Alex Rogozhnikov'


class SklearnBase(object):
    """
    SklearnBase is base for sklearn-like models.
    All attributes will be returned for base estimator

    Parameters:
    -----------
    :param sklearn.BaseEstimator clf: your estimator, which will be used for training

    """

    __metaclass__ = ABCMeta

    def __init__(self, clf):
        self.clf = clf

    def _fit(self, X, y, sample_weight=None, **kwargs):
        """
        Train the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self

        .. note:: if sklearn classifier doesn't support *sample_weight* then put *sample_weight=None*,
        else exception will be thrown.
        """
        if sample_weight is None:
            self.clf.fit(self._get_features(X), y, **kwargs)
        else:
            self.clf.fit(self._get_features(X), y, sample_weight=sample_weight, **kwargs)
        return self

    def predict(self, X):
        """
        Predict labels for estimators and values for regressors for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples] with labels/values
        """
        return self.clf.predict(self._get_features(X))

    def __getattr__(self, name):
        if name in ['__getstate__', '__setstate__']:
            raise AttributeError()
        # Those methods not defined here will be used directly from classifier
        base = self.clf
        return base.__getattribute__(name)


class SklearnClassifier(SklearnBase, Classifier):
    """
    SklearnClassifier is wrapper on sklearn-like **estimators**

    Parameters:
    -----------
    :param sklearn.BaseEstimator clf: your classifier, which will be used for training
    :param features: features used in training
    :type features: list[str] or None
    """

    def __init__(self, clf, features=None):
        if isinstance(clf, Classifier):
            raise ValueError('Base class should be simply sklearn classifier, not descendant of Classifier')
        SklearnBase.__init__(self, clf=clf)
        Classifier.__init__(self, features=features)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in model

        .. note:: parameters of base estimator can be accessed (for example param `depth`)
            by both *depth* and *clf__depth*.
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

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Train the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self

        .. note:: if sklearn classifier doesn't support *sample_weight*, put *sample_weight=None*,
            otherwise exception will be thrown.
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        self._set_classes(y)
        return self._fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict_proba(self, X):
        """
        Predict probabilities for each class label on dataset

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: numpy.array of shape [n_samples, n_classes] with probabilities
        """
        return self.clf.predict_proba(self._get_features(X))

    def staged_predict_proba(self, X):
        """
        Predicts probabilities on each stage

        :return: iterator
        """
        return self.clf.staged_predict_proba(self._get_features(X))


class SklearnRegressor(SklearnBase, Regressor):
    """
    SklearnClassifier is wrapper on sklearn-like regressors

    Parameters:
    -----------
    :param sklearn.BaseEstimator clf: your classifier, which will be used for training
    :param features: features used in training
    :type features: list[str] or None
    """

    def __init__(self, clf, features=None):
        if isinstance(clf, Regressor):
            raise ValueError('Base class should be simply sklearn classifier, not descendant of Regressor')
        SklearnBase.__init__(self, clf=clf)
        Regressor.__init__(self, features=features)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in model

        .. note:: Access to all parameters of base estimator can be done by (for example param `depth`) *model.clf__depth*.
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

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Train the classifier

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self

        .. note:: if sklearn classifier doesn't support *sample_weight* then put *sample_weight=None*,
        else exception will be thrown.
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)
        return self._fit(X, y, sample_weight=sample_weight, **kwargs)

    def staged_predict(self, X):
        """
        Predicts regression target at each stage for X.
        This method allows monitoring of error after each stage.

        :return: iterator
        """
        return self.clf.staged_predict(self._get_features(X))
