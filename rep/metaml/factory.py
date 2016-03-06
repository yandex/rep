"""
**Factory** provides convenient way to train several classifiers on the same dataset.
These classifiers can be trained one-by-one in a single thread, or simultaneously
with IPython cluster or in several threads.

Also :class:`Factory` allows comparison of several classifiers (predictions of which can be computed again in parallel).

"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import time

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from ..report import classification, regression
from ..estimators.interface import Classifier, Regressor
from ..estimators.sklearn import SklearnClassifier, SklearnRegressor
from . import utils

__author__ = 'Tatiana Likhomanenko'

__all__ = ['ClassifiersFactory', 'RegressorsFactory']


class AbstractFactory(OrderedDict):
    """
    Factory provides interface to train simultaneously several estimators (classifiers or regressors).
    Later their quality can be compared.
    """

    __metaclass__ = ABCMeta

    def fit(self, X, y, sample_weight=None, parallel_profile=None, features=None):
        """
        Train all estimators on the same data.

        :param X: pandas.DataFrame of shape [n_samples, n_features] with features
        :param y: array-like of shape [n_samples] with labels of samples
        :param sample_weight: weights of events,
               array-like of shape [n_samples] or None if all weights are equal
        :param features: features to train estimators
            If None, estimators will be trained on `estimator.features`
        :type features: None or list[str]
        :param parallel_profile: profile of parallel execution system or None
        :type parallel_profile: None or str

        :return: self
        """
        if features is not None:
            for name, estimator in self.items():
                if estimator.features is not None:
                    print('Overwriting features of estimator ' + name)
                self[name].set_params(features=features)

        start_time = time.time()
        result = utils.map_on_cluster(parallel_profile, train_estimator, list(self.keys()), list(self.values()),
                                      [X] * len(self), [y] * len(self), [sample_weight] * len(self))
        for status, data in result:
            if status == 'success':
                name, estimator, spent_time = data
                self[name] = estimator
                print('model {:12} was trained in {:.2f} seconds'.format(name, spent_time))
            else:
                print('Problem while training on the node, report:\n', data)

        print("Totally spent {:.2f} seconds on training".format(time.time() - start_time))
        return self

    def fit_lds(self, lds, parallel_profile=None, features=None):
        """
        Fit all estimators on the same dataset.

        :param LabeledDataStorage lds: dataset
        :param features: features to train estimators
            If None, estimators will be trained on `estimator.features`
        :param parallel_profile: profile of parallel execution system or None
        :type parallel_profile: None or str

        :return: self
        """
        X, y, sample_weight = lds.get_data(features), lds.get_targets(), lds.get_weights()
        return self.fit(X, y, sample_weight=sample_weight, parallel_profile=parallel_profile, features=features)

    @abstractmethod
    def predict(self, X, parallel_profile=None):
        """
        Predict labels (or values for regressors) for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param parallel_profile: profile of parallel execution system or None
        :type parallel_profile: None or str

        :rtype: OrderedDict[numpy.array of shape [n_samples] with integer labels (or values)]
        """
        pass

    @abstractmethod
    def test_on_lds(self, lds):
        """
        Prepare report for factory (comparison of all models).

        :param LabeledDataStorage lds: data
        :rtype: rep.report.classification.ClassificationReport or rep.report.regression.RegressionReport
        """
        pass

    def test_on(self, X, y, sample_weight=None):
        """
        Prepare report for factory (comparison of all models).

        :param X: pandas.DataFrame of shape [n_samples, n_features] with features
        :param y: numpy.array of shape [n_samples] with targets
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :rtype: rep.report.classification.ClassificationReport or rep.report.regression.RegressionReport
        """
        from ..data import LabeledDataStorage
        return self.test_on_lds(LabeledDataStorage(X, target=y, sample_weight=sample_weight))


class ClassifiersFactory(AbstractFactory):
    """
    Factory provides training of several classifiers in parallel.
    Quality of trained classifiers can be compared.
    """

    def add_classifier(self, name, classifier):
        """
        Add classifier to factory.
        Automatically wraps classifier with :class:`SklearnClassifier`

        :param str name: unique name for classifier.
            If name coincides with one already used, the old classifier will be replaced by one passed.
        :param  classifier: classifier object

            .. note:: if type == sklearn.base.BaseEstimator, then features=None is used,
                to specify features used by classifier, wrap it with :class:`SklearnClassifier`

        :type classifier: sklearn.base.BaseEstimator or estimators.interface.Classifier
        """
        if isinstance(classifier, Classifier):
            self[name] = classifier
        elif isinstance(classifier, BaseEstimator) and isinstance(classifier, ClassifierMixin):
            self[name] = SklearnClassifier(classifier)
        else:
            raise NotImplementedError(
                'Supports only instances of sklearn.base.BaseEstimator or rep.estimators.interface.Classifier')

    def predict(self, X, parallel_profile=None):
        """
        Predict labels for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param parallel_profile: profile for IPython cluster
        :type parallel_profile: None or str

        :rtype: OrderedDict[numpy.array of shape [n_samples] with integer labels]
        """
        return self._predict_method(X, parallel_profile=parallel_profile, prediction_type='classification')

    def predict_proba(self, X, parallel_profile=None):
        """
        Predict probabilities for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param parallel_profile: profile
        :type parallel_profile: None or str

        :rtype: OrderedDict[numpy.array of shape [n_samples] with float predictions]
        """
        return self._predict_method(X, parallel_profile=parallel_profile, prediction_type='classification-proba')

    def _predict_method(self, X, parallel_profile=None, prediction_type='classification'):
        """
        Predict probabilities for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param parallel_profile: profile
        :type parallel_profile: None or str
        :param str prediction_type: 'classification' or 'regression' or 'classification-proba'

        :rtype: OrderedDict[numpy.array of shape [n_samples] with float predictions]
        """
        predictions = OrderedDict()

        start_time = time.time()
        result = utils.map_on_cluster(parallel_profile, predict_estimator, list(self.keys()), list(self.values()), [X] * len(self),
                                      [prediction_type] * len(self))

        for status, data in result:
            if status == 'success':
                name, prob, spent_time = data
                predictions[name] = prob
                print('data was predicted by {:12} in {:.2f} seconds'.format(name, spent_time))
            else:
                print('Problem while predicting on the node, report:\n', data)

        print("Totally spent {:.2f} seconds on prediction".format(time.time() - start_time))
        return predictions

    def staged_predict_proba(self, X):
        """
        Predict probabilities on each stage (attention: returns dictionary of generators)

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: dict[iterator]
        """
        generators_dict = OrderedDict()
        for name, classifier in self.items():
            try:
                generators_dict[name] = classifier.staged_predict_proba(X)
            except AttributeError:
                pass
        return generators_dict

    def test_on_lds(self, lds):
        """
        Prepare report for factory of estimators

        :param LabeledDataStorage lds: data
        :rtype: rep.report.classification.ClassificationReport
        """
        return classification.ClassificationReport(self, lds)


class RegressorsFactory(AbstractFactory):
    """
    Factory provides training of several classifiers in parallel.
    Quality of trained regressors can be compared.
    """

    def add_regressor(self, name, regressor):
        """
        Add regressor to factory

        :param str name: unique name for regressor.
            If name coincides with one already used, the old regressor will be replaced by one passed.
        :param regressor: regressor object

            .. note:: if type == sklearn.base.BaseEstimator, then features=None is used,
                to specify features used by regressor, wrap it first with :class:`SklearnRegressor`
        :type regressor: sklearn.base.BaseEstimator or estimators.interface.Regressor

        """
        if isinstance(regressor, Regressor):
            self[name] = regressor
        elif isinstance(regressor, BaseEstimator) and isinstance(regressor, RegressorMixin):
            self[name] = SklearnRegressor(regressor)
        else:
            raise NotImplementedError(
                'Supports only instances of sklearn.base.BaseEstimator or rep.estimators.interface.Regressor')

    def predict(self, X, parallel_profile=None):
        """
        Predict values for all events in dataset.

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param parallel_profile: profile
        :type parallel_profile: None or name of profile to parallelize computations.

        :rtype: OrderedDict[numpy.array of shape [n_samples] with float values]
        """
        predictions = OrderedDict()

        start_time = time.time()
        result = utils.map_on_cluster(parallel_profile, predict_estimator, list(self.keys()), list(self.values()), [X] * len(self),
                                      ['regression'] * len(self))

        for status, data in result:
            if status == 'success':
                name, values, spent_time = data
                predictions[name] = values
                print('data was predicted by {:12} in {:.2f} seconds'.format(name, spent_time))
            else:
                print('Problem while predicting on the node, report:\n', data)

        print("Totally spent {:.2f} seconds on prediction".format(time.time() - start_time))
        return predictions

    def staged_predict(self, X):
        """
        Predicts probabilities on each stage

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :rtype: dict[iterator]
        """
        generators_dict = OrderedDict()
        for name, regressor in self.items():
            try:
                generators_dict[name] = regressor.staged_predict(X)
            except AttributeError:
                pass
        return generators_dict

    def test_on_lds(self, lds):
        """
        Report for factory of estimators

        :param LabeledDataStorage lds: data
        :rtype: rep.report.regression.RegressionReport
        """
        return regression.RegressionReport(self, lds)


def train_estimator(name, estimator, X, y, sample_weight=None):
    """
    Supplementary function.
    Trains estimator on a separate node (or in a separate thread)

    :param str name: classifier name
    :param estimator: estimator
    :type estimator: Classifier or Regressor
    :param X: pandas.DataFrame of shape [n_samples, n_features]
    :param y: labels of events - array-like of shape [n_samples]
    :param sample_weight: weight of events,
           array-like of shape [n_samples] or None if all weights are equal

    :return: ('success', (name (str), estimator (Classifier or Regressor), time (int) )) or
             ('fail', (name (str), pid (int), socket (int), error (Exception) ))
    """
    try:
        start = time.time()
        if sample_weight is None:
            estimator.fit(X, y)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)
        return 'success', (name, estimator, time.time() - start)
    except Exception as e:
        import socket
        import os

        pid = os.getpid()
        hostname = socket.gethostname()
        return 'fail', (name, pid, hostname, e)


def predict_estimator(name, estimator, X, prediction_type='classification'):
    """
    Supplementary function.
    Builds predictions for one estimator on a separate node (or in a separate thread)

    :param str name: classifier name
    :param estimator: estimator
    :type estimator: Classifier or Regressor
    :param X: pandas.DataFrame of shape [n_samples, n_features]
    :param str prediction_type: 'classification' or 'regression' or 'classification-proba'

    :return: ('success', (name (str), probabilities (numpy.array), time (int) )) or
             ('fail', (name (str), pid (int), socket (int), error (Exception) ))

    """
    try:
        start = time.time()
        if prediction_type == 'classification':
            prediction = estimator.predict(X)
        elif prediction_type == 'classification-proba':
            prediction = estimator.predict_proba(X)
        elif prediction_type == 'regression':
            prediction = estimator.predict(X)
        else:
            raise NotImplementedError("Unknown problem type: {}".format(prediction_type))
        return 'success', (name, prediction, time.time() - start)
    except Exception as e:
        import socket
        import os
        pid = os.getpid()
        hostname = socket.gethostname()
        return 'fail', (name, pid, hostname, e)
