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
from sklearn.preprocessing.data import StandardScaler
from rep.test.test_estimators import check_classifier, check_regression, check_params, \
    check_classification_reproducibility
from rep.test.test_estimators import generate_classification_data
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from rep.estimators.sklearn import SklearnClassifier
from rep.estimators.theanets import TheanetsClassifier, TheanetsRegressor
import climate
from tests import known_failure, retry_if_fails

climate.enable_default_logging(default_level='ERROR')

__author__ = 'Lisa Ignatyeva, Tatiana Likhomanenko, Alex Rogozhnikov'

classifier_params = {
    'has_staged_pp': False,
    'has_importances': False,
    'supports_weight': True,
}

regressor_params = {
    'has_staged_predictions': False,
    'has_importances': False,
    'supports_weight': True,
}


@known_failure
def test_exp_failure():
    assert 0 == 1


@retry_if_fails
def test_theanets_params():
    check_params(TheanetsClassifier, layers=[1, 2], scaler=False, trainers=[{}, {'algo': 'nag', 'learning_rate': 0.1}])
    check_params(TheanetsRegressor, layers=[1, 2], scaler=False, trainers=[{}, {'algo': 'nag', 'learning_rate': 0.1}])


@retry_if_fails
def test_pretrain():
    clf = TheanetsClassifier(layers=[5, 5], trainers=[{'algo': 'pretrain', 'learning_rate': 0.1},
                                                      {'algo': 'nag', 'learning_rate': 0.1}])
    check_classifier(clf, **classifier_params)


@retry_if_fails
def test_theanets_configurations():
    check_classifier(
        TheanetsClassifier(layers=[13], scaler=False,
                           trainers=[{'algo': 'nag', 'learning_rate': 0.1}]),
        **classifier_params)
    check_classifier(
        TheanetsClassifier(layers=[5, 5], scaler='minmax',
                           trainers=[{'algo': 'adadelta', 'learning_rate': 0.1}]),
        **classifier_params)


@retry_if_fails
def test_theanets_regression():
    check_regression(TheanetsRegressor(layers=[11],
                                       trainers=[{'algo': 'rmsprop', 'learning_rate': 0.1}]),
                     **regressor_params)
    check_regression(TheanetsRegressor(scaler=StandardScaler()), **regressor_params)


@known_failure
def test_theanets_partial_fit():
    clf_complete = TheanetsClassifier(layers=[2], trainers=[{'algo': 'rmsprop', 'learning_rate': 0.1},
                                                            {'algo': 'rprop', 'learning_rate': 0.1}])
    clf_partial = TheanetsClassifier(layers=[2], trainers=[{'algo': 'rmsprop', 'learning_rate': 0.1}])
    X, y, sample_weight = generate_classification_data()
    import numpy
    numpy.random.seed(43)
    clf_complete.fit(X, y)
    clf_partial.fit(X, y)
    clf_partial.partial_fit(X, y, algo='rprop', learning_rate=0.1)

    assert clf_complete.trainers == clf_partial.trainers, 'trainers not saved in partial fit'

    auc_complete = roc_auc_score(y, clf_complete.predict_proba(X)[:, 1])
    auc_partial = roc_auc_score(y, clf_partial.predict_proba(X)[:, 1])

    # Known fail of theanets
    assert auc_complete == auc_partial, 'same networks return different results'


@known_failure
def test_theanets_reproducibility():
    clf = TheanetsClassifier(trainers=[{'algo': 'nag', 'min_improvement': 0.1}])
    X, y, _ = generate_classification_data()
    import numpy
    numpy.random.seed(43)
    check_classification_reproducibility(clf, X, y)


@retry_if_fails
def test_theanets_simple_stacking():
    base_tnt = TheanetsClassifier(trainers=[{'min_improvement': 0.1}])
    base_bagging = BaggingClassifier(base_estimator=base_tnt, n_estimators=3)
    check_classifier(SklearnClassifier(clf=base_bagging), **classifier_params)


@retry_if_fails
def test_theanets_multiclassification():
    check_classifier(TheanetsClassifier(trainers=[{'min_improvement': 0.1, 'learning_rate': 0.1}]), n_classes=4,
                     **classifier_params)


def test_theanets_multi_regression():
    check_regression(TheanetsRegressor(layers=[13], trainers=[{'algo': 'rmsprop', 'min_improvement': 0.1}]),
                     n_targets=3, **regressor_params)
