from __future__ import division, print_function, absolute_import
import time
import os.path
import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from rep.metaml._cache import CacheHelper
from rep.metaml.cache import CacheClassifier, CacheRegressor, cache_helper
from rep.test.test_estimators import generate_classification_data, check_classifier, check_regression

__author__ = 'Alex Rogozhnikov'


def test_cache_helper():
    cache = CacheHelper(folder='./.cache/rep', expiration_in_seconds=1000)
    cache.store_in_cache('first', 'hash', 24)
    cache.store_in_cache('first', 'hash', 42)
    cache.store_in_cache('second', 'hash', 45)
    assert cache.get_from_cache('first', 'hash') == (True, 42)
    assert cache.get_from_cache('first', 'wrong_hash')[0] == False
    cache.clear_cache()
    assert cache.get_from_cache('first', 'hash')[0] == False
    assert cache.get_from_cache('first', 'wrong_hash')[0] == False
    cache.clear_cache()


def test_cache_expiration(folder='./.cache/rep'):
    cache = CacheHelper(folder=folder, expiration_in_seconds=1000)
    cache.store_in_cache('first', 'hash', 42)
    assert cache.get_from_cache('first', 'hash') == (True, 42)
    for file_name in os.listdir(cache.folder):
        new_time = datetime.datetime.now() - datetime.timedelta(seconds=10)
        new_time = time.mktime(new_time.timetuple())
        file_path = os.path.join(cache.folder, file_name)
        os.utime(file_path, (new_time, new_time))
    # should be able to find
    assert cache.get_from_cache('first', 'hash') == (True, 42)

    for file_name in os.listdir(cache.folder):
        new_time = datetime.datetime.now() - datetime.timedelta(seconds=2000)
        new_time = time.mktime(new_time.timetuple())
        file_path = os.path.join(cache.folder, file_name)
        os.utime(file_path, (new_time, new_time))

    # should not be able to find
    assert cache.get_from_cache('first', 'hash')[0] == False
    cache.clear_cache()


def test_cache_classifier():
    cache_helper.clear_cache()

    for Wrapper, Model in [(CacheClassifier, LogisticRegression), (CacheRegressor, LinearRegression)]:
        X, y, weights = generate_classification_data(n_classes=2)
        clf = Wrapper('first', Model()).fit(X, y)
        assert clf._used_cache == False
        clf = Wrapper('first', Model()).fit(X + 0, y + 0)
        assert clf._used_cache == True
        # changed name
        clf = Wrapper('second', Model()).fit(X, y)
        assert clf._used_cache == False
        # changed data
        X_new = X.copy()
        X_new.iloc[0, 0] += 1
        clf = Wrapper('first', Model()).fit(X_new, y)
        assert clf._used_cache == False
        # changed labels
        y_new = y.copy()
        y_new[0] += 1
        clf = Wrapper('first', Model()).fit(X, y_new)
        assert clf._used_cache == False
        # added weights
        clf = Wrapper('first', Model()).fit(X, y, sample_weight=None)
        assert clf._used_cache == False
        # changed parameters
        clf = Wrapper('first', Model(n_jobs=2)).fit(X, y)
        assert clf._used_cache == False
        # fitting previous once again. Checking that overwriting is correct.
        clf = Wrapper('first', Model(n_jobs=2)).fit(X, y)
        assert clf._used_cache == True

    cache_helper.clear_cache()


def test_models():
    for _ in range(3):
        clf = CacheClassifier('clf', SGDClassifier(loss='log'))
        check_classifier(clf, has_staged_pp=False, has_importances=False)

        reg = CacheRegressor('reg', SGDRegressor())
        check_regression(reg, has_staged_predictions=False, has_importances=False)
    cache_helper.clear_cache()
