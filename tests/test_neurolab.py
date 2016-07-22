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
from rep.test.test_estimators import check_classifier, check_regression, generate_classification_data, \
    check_params, check_classification_reproducibility
from sklearn.ensemble import BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier
from rep.estimators.neurolab import NeurolabClassifier, NeurolabRegressor
import rep
import neurolab as nl

__author__ = 'Sterzhanov Vladislav'

N_EPOCHS2 = 20
N_EPOCHS4 = 60
N_EPOCHS_REGR = 10

classifier_params = {
    'has_staged_pp': False,
    'has_importances': False,
    'supports_weight': False,
}

regressor_params = {
    'has_staged_predictions': False,
    'has_importances': False,
    'supports_weight': False,
}


def test_neurolab_params():
    check_params(NeurolabClassifier, layers=[1, 2], epochs=5, trainf='blah', cn=2, omnomnom=4)
    check_params(NeurolabRegressor, layers=[1, 2], epochs=5, trainf='blah', cn=2, omnomnom=4)


def test_neurolab_single_classification():
    check_classifier(NeurolabClassifier(layers=[], epochs=N_EPOCHS2, trainf=None),
                     **classifier_params)
    check_classifier(NeurolabClassifier(layers=[2], epochs=N_EPOCHS2),
                     **classifier_params)
    check_classifier(NeurolabClassifier(layers=[1, 1], epochs=N_EPOCHS2),
                     **classifier_params)


def test_partial_fit():
    clf = NeurolabClassifier(layers=[4, 5], epochs=2, trainf=nl.train.train_gd)
    X, y, _ = generate_classification_data()
    clf.fit(X, y)
    clf.partial_fit(X[:2], y[:2])


def test_neurolab_regression():
    check_regression(NeurolabRegressor(layers=[1], epochs=N_EPOCHS_REGR), **regressor_params)


def test_neurolab_reproducibility():
    clf = NeurolabClassifier(layers=[4, 5], epochs=2, trainf=nl.train.train_gd)
    X, y, _ = generate_classification_data()
    check_classification_reproducibility(clf, X, y)


def test_neurolab_multiclassification():
    check_classifier(NeurolabClassifier(layers=[10], epochs=N_EPOCHS4, trainf=nl.train.train_rprop),
                     n_classes=4, **classifier_params)


def test_neurolab_multi_regression():
    check_regression(NeurolabRegressor(layers=[10], epochs=N_EPOCHS_REGR),
                     n_targets=3, **regressor_params)


def test_neurolab_stacking():
    base_nlab = NeurolabClassifier(layers=[], epochs=N_EPOCHS2 * 2, trainf=nl.train.train_rprop)
    base_bagging = BaggingClassifier(base_estimator=base_nlab, n_estimators=3)
    check_classifier(SklearnClassifier(clf=base_bagging), **classifier_params)


def test_neurolab_classification_types():
    import pandas as pd
    for net_type in rep.estimators.neurolab.NET_TYPES.keys():
        try:
            clf = NeurolabClassifier(net_type=net_type, epochs=2)
            ds = pd.DataFrame()
            ds['feature1'] = [0, 1, 2, 3, 4, 5]
            ds['feature2'] = [5, 7, 2, 4, 7, 9]
            ds['y'] = [0, 0, 0, 1, 1, 1]
            clf.fit(ds[['feature1', 'feature2']] / 10., ds['y'])
            _ = clf.predict_proba(ds[['feature1', 'feature2']] / 10.)
            print(net_type, 'is ok')
        except Exception as e:
            print(net_type, 'FAILED', e)


def test_neurolab_regression_types():
    import pandas as pd
    for net_type in rep.estimators.neurolab.NET_TYPES.keys():
        try:
            clf = NeurolabRegressor(net_type=net_type, epochs=2)
            ds = pd.DataFrame()
            ds['feature1'] = [0, 1, 2, 3, 4, 5]
            ds['feature2'] = [5, 7, 2, 4, 7, 9]
            ds['y'] = [0, 0, 0, 1, 1, 1]
            clf.fit(ds[['feature1', 'feature2']] / 10., ds['y'])
            _ = clf.predict(ds[['feature1', 'feature2']] / 10.)
            print(net_type, 'is ok')
        except Exception as e:
            print(net_type, 'FAILED', e)
