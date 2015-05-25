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


def test_pybrain_classification():
    check_classifier(PyBrainClassifier(), has_staged_pp=False, has_importances=False, supports_weight=False)
    check_classifier(PyBrainClassifier(layers=[10, 10]), has_staged_pp=False, has_importances=False, supports_weight=False)


def test_pybrain_Linear():
    check_classifier(PyBrainClassifier(layers=[10], hiddenclass=['LinearLayer']),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
    check_regression(PyBrainRegressor(layers=[10], hiddenclass=['LinearLayer']),
                     has_staged_predictions=False, has_importances=False, supports_weight=False)


def test_pybrain_MDLSTM():
    check_classifier(PyBrainClassifier(layers=[10], hiddenclass=['MDLSTMLayer']),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
    check_regression(PyBrainRegressor(layers=[10], hiddenclass=['MDLSTMLayer']),
                     has_staged_predictions=False, has_importances=False, supports_weight=False)


def test_pybrain_SoftMax():
    check_classifier(PyBrainClassifier(layers=[10], hiddenclass=['SoftmaxLayer']),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
    check_regression(PyBrainRegressor(layers=[10], hiddenclass=['SoftmaxLayer']),
                     has_staged_predictions=False, has_importances=False, supports_weight=False)


def test_pybrain_Tanh():
    check_classifier(PyBrainClassifier(layers=[10], hiddenclass=['TanhLayer']),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
    check_regression(PyBrainRegressor(layers=[10], hiddenclass=['TanhLayer']),
                     has_staged_predictions=False, has_importances=False, supports_weight=False)


def test_pybrain_rprop():
    check_classifier(PyBrainClassifier(use_rprop=True), has_staged_pp=False, has_importances=False, supports_weight=False)


def test_pybrain_multiclassification():
    check_classifier(PyBrainClassifier(), has_staged_pp=False, has_importances=False, supports_weight=False, n_classes=4)


def test_pybrain_regression():
    check_regression(PyBrainRegressor(), has_staged_predictions=False, has_importances=False, supports_weight=False)


def test_simple_stacking_pybrain():
    base_pybrain = PyBrainClassifier()
    check_classifier(SklearnClassifier(clf=BaggingClassifier(base_estimator=base_pybrain, n_estimators=3)),
                     has_staged_pp=False, has_importances=False, supports_weight=False)
