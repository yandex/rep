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
from rep.test.test_estimators import check_classifier, check_regression
from rep.estimators.pybrain import PyBrainClassifier
from rep.estimators.pybrain import PyBrainRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from rep.estimators import SklearnClassifier


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


def test_pybrain_classification():
    check_classifier(PyBrainClassifier(), **classifier_params)
    check_classifier(PyBrainClassifier(layers=[]), **classifier_params)
    check_classifier(PyBrainClassifier(layers=[10, 10]), **classifier_params)


def test_pybrain_Linear_MDLSTM():
    check_classifier(PyBrainClassifier(layers=[10, 2], hiddenclass=['LinearLayer', 'MDLSTMLayer']), **classifier_params)
    check_regression(PyBrainRegressor(layers=[10, 2], hiddenclass=['LinearLayer', 'MDLSTMLayer']), **regressor_params)


def test_pybrain_SoftMax():
    check_classifier(PyBrainClassifier(layers=[10], hiddenclass=['SoftmaxLayer']), **classifier_params)
    check_regression(PyBrainRegressor(layers=[10], hiddenclass=['SoftmaxLayer']), **regressor_params)


def test_pybrain_Tanh():
    check_classifier(PyBrainClassifier(layers=[10], hiddenclass=['TanhLayer']), **classifier_params)
    check_regression(PyBrainRegressor(layers=[10], hiddenclass=['TanhLayer']), **regressor_params)


def test_pybrain_rprop():
    check_classifier(PyBrainClassifier(use_rprop=True), **classifier_params)


def test_pybrain_multi_classification():
    check_classifier(PyBrainClassifier(), n_classes=4, **classifier_params)


def test_pybrain_regression():
    check_regression(PyBrainRegressor(), **regressor_params)


def test_pybrain_multi_regression():
    check_regression(PyBrainRegressor(), n_targets=4, **regressor_params)


def test_simple_stacking_pybrain():
    base_pybrain = PyBrainClassifier()
    base_bagging = BaggingClassifier(base_estimator=base_pybrain, n_estimators=3)
    check_classifier(SklearnClassifier(clf=base_bagging), **classifier_params)
