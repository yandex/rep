# Copyright 2014-2015 Yandex LLC and contributors <https://yandex.com/>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# <http://www.apache.org/licenses/LICENSE-2.0>
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import division, print_function, absolute_import

import six

if six.PY3:
    # PyBrain doesn't support python3
    import nose

    raise nose.SkipTest

import numpy
from rep.test.test_estimators import check_classifier, check_regression, check_params, \
    generate_classification_data, check_classification_reproducibility
from rep.estimators.pybrain import PyBrainClassifier, PyBrainRegressor
from sklearn.ensemble import BaggingClassifier
from rep.estimators import SklearnClassifier
from . import known_failure

__author__ = 'Artem Zhirokhov'

classifier_params = {
    'has_staged_pp': False,
    'has_importances': False,
    'supports_weight': False
}

regressor_params = {
    'has_staged_predictions': False,
    'has_importances': False,
    'supports_weight': False
}


def test_pybrain_params():
    check_params(PyBrainClassifier, layers=[1, 2], epochs=5, use_rprop=True, hiddenclass=['LinearLayer'])
    check_params(PyBrainRegressor, layers=[1, 2], epochs=5, etaplus=1.3, hiddenclass=['LinearLayer'], learningrate=0.1)


def test_pybrain_classification():
    clf = PyBrainClassifier(epochs=2)
    check_classifier(clf, **classifier_params)
    check_classifier(PyBrainClassifier(epochs=-1, continue_epochs=1, layers=[]), **classifier_params)
    check_classifier(PyBrainClassifier(epochs=2, layers=[5, 2]), **classifier_params)


@known_failure
def test_pybrain_reproducibility():
    # This test fails. Because PyBrain can't reproduce training.
    X, y, _ = generate_classification_data()
    clf1 = PyBrainClassifier(layers=[4], epochs=2).fit(X, y)
    clf2 = PyBrainClassifier(layers=[4], epochs=2).fit(X, y)
    print(clf1.predict_proba(X) - clf2.predict_proba(X))
    assert numpy.allclose(clf1.predict_proba(X), clf2.predict_proba(X)), 'different predicitons'
    check_classification_reproducibility(clf1, X, y)


def test_pybrain_Linear_MDLSTM():
    check_classifier(PyBrainClassifier(epochs=2, layers=[10, 2], hiddenclass=['LinearLayer', 'MDLSTMLayer']),
                     **classifier_params)
    check_regression(PyBrainRegressor(epochs=3, layers=[10, 2], hiddenclass=['LinearLayer', 'MDLSTMLayer']),
                     **regressor_params)


def test_pybrain_SoftMax_Tanh():
    check_classifier(PyBrainClassifier(epochs=10, layers=[5, 2], hiddenclass=['TanhLayer', 'SoftmaxLayer'],
                                       use_rprop=True),
                     **classifier_params)
    check_regression(
        PyBrainRegressor(epochs=2, layers=[10, 5, 2], hiddenclass=['TanhLayer', 'SoftmaxLayer', 'TanhLayer']),
        **regressor_params)


def pybrain_test_partial_fit():
    clf = PyBrainClassifier(layers=[4], epochs=2)
    X, y, _ = generate_classification_data()
    clf.partial_fit(X, y)
    clf.partial_fit(X[:2], y[:2])


def test_pybrain_multi_classification():
    check_classifier(PyBrainClassifier(), n_classes=4, **classifier_params)


def test_pybrain_regression():
    check_regression(PyBrainRegressor(), **regressor_params)


def test_pybrain_multi_regression():
    check_regression(PyBrainRegressor(), n_targets=4, **regressor_params)


def test_simple_stacking_pybrain():
    base_pybrain = PyBrainClassifier(epochs=2)
    base_bagging = BaggingClassifier(base_estimator=base_pybrain, n_estimators=3)
    check_classifier(SklearnClassifier(clf=base_bagging), **classifier_params)
