"""

In many cases training a classification/regression takes hours.
To avoid retraining at each step, one can store trained classifier in a file,
and later load trained model.

However, in this case user should care about situations when something changed in the pipeline
(for instance, train/test splitting) manually.

Cache estimators are lazy way to store trained model.
After training, classifier/regressor is stored in the file under specific name (which was passed in constructor).

On the next runs following conditions are checked:

* model has the same name
* model trained has exactly same parameters
* model is trained using exactly the same data
* stored copy in not too old (10 days by default)

If all the conditions satisfied, stored copy is loaded, otherwise classifier/regressor is fitted.

Example of usage
----------------

:class:`CacheClassifier` and :class:`CacheRegressor` work as meta-estimators

>>> from rep.estimators import XGBoostClassifier
>>> from rep.metaml import FoldingClassifier
>>> from rep.metaml.cache import CacheClassifier
>>> clf = CacheClassifier('xgboost folding', FoldingClassifier(XGBoostClassifier(), n_folds=3))
>>> # this works normally
>>> clf.fit(X, y, sample_weight)
>>> clf.predict_proba(testX)

However in the following situation:

>>> clf = FoldingClassifier(CacheClassifier('xgboost', XGBoostClassifier()))

cache is not going to work, because for each fold a copy of classifier is created.
Each time after looking at cache, a version with same parameters, but different data will be found.

So, every time stored copy will be erased and a new one saved.


By default, cache is stored in '.cache/rep' subfolder of project directory (where the ipython notebook is placed).
To change parameters of caching use:

>>> import rep.metaml.cache
>>> from rep.metaml._cache import CacheHelper
>>> rep.metaml.cache.cache_helper = CacheHelper(folder, expiration_in_seconds)
>>> # to delete all cached items, use:
>>> rep.metaml.cache.cache_helper.clear_cache()

"""

from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

from ..estimators.interface import Classifier, Regressor
from ._cache import CacheHelper
from ..estimators import SklearnClassifier, SklearnRegressor
import hashlib
from six.moves import cPickle
from sklearn.base import ClassifierMixin, RegressorMixin

# To change cache parameters use
cache_helper = CacheHelper(folder='./.cache/rep',
                           expiration_in_seconds=10 * 24 * 60 * 60,  # 10 days
                           )

__all__ = ['CacheClassifier', 'CacheRegressor']


class CacheBase(object):
    def __init__(self, name, clf, features=None):
        """
        Cache {estimator} allows to save trained models in lazy way.
        Useful when training {estimator} takes much time.

        On the next run, stored model in cache will be used instead of fitting again.

        :param name: unique name of classifier (to be used in storing)
        :param sklearn.BaseEstimator clf: your estimator, which will be used for training
        :param features: features to use in training.
        """
        self.features = features
        self.name = name
        self.clf = clf
        self._used_cache = None

    def _fit(self, X, y, **kwargs):
        """
        Train the {estimator}

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: target of training - array-like of shape [n_samples]
        :param sample_weight: weights of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: self
        """
        X = self._get_features(X)
        parameters = self.clf, X, y, sorted(kwargs)
        hash_value = hashlib.sha1(cPickle.dumps(parameters)).hexdigest()
        is_found, trained_clf = cache_helper.get_from_cache(self.name, hash_value)
        if is_found:
            print('Found a trained copy, skipped fitting.')
            self.clf = trained_clf
            self._used_cache = True
        else:
            print('Not found in the cache (previous version may have expired). Fitting.')
            self.clf.fit(X, y, **kwargs)
            cache_helper.store_in_cache(self.name, hash_value, self.clf)
            self._used_cache = False
        return self

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Parameters of base estimator can be accessed (for example param `depth`) by both *depth* and *clf__depth*.

        :param dict params: parameters to set in model
        """
        params_for_clf = {}
        for name, value in params.items():
            if name == 'features':
                self.features = value
            elif name == 'name':
                self.name = value
            elif name == 'clf':
                self.clf = value
            elif name.startswith('clf__'):
                params_for_clf[name[5:]] = value
            else:
                params_for_clf[name] = value
        self.clf.set_params(**params_for_clf)


class CacheClassifier(CacheBase, SklearnClassifier):
    def __init__(self, name, clf, features=None):
        if not isinstance(clf, ClassifierMixin):
            raise ValueError('passed model should be derived from ClassifierMixin!')

        CacheBase.__init__(self, name=name, clf=clf, features=features)
        Classifier.__init__(self, features=features)

    __init__.__doc__ = CacheBase.__init__.__doc__.format(estimator='classifier')


class CacheRegressor(CacheBase, SklearnRegressor):
    def __init__(self, name, clf, features=None):
        if not isinstance(clf, RegressorMixin):
            raise ValueError('passed model should be derived from RegressorMixin!')

        CacheBase.__init__(self, name=name, clf=clf, features=features)
        Regressor.__init__(self, features=features)

    __init__.__doc__ = CacheBase.__init__.__doc__.format(estimator='regressor')
