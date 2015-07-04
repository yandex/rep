from __future__ import division, print_function, absolute_import

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy

from rep.data import LabeledDataStorage
from rep.metaml import ClassifiersFactory
from six.moves import cPickle
from rep.report import ClassificationReport
from rep.report.metrics import significance
from rep.test.test_estimators import generate_classification_data
from rep.report.metrics import RocAuc


__author__ = 'Tatiana Likhomanenko'

# TODO testing of right-classification part of estimators


def test_factory():
    factory = ClassifiersFactory()
    try:
        from rep.estimators.tmva import TMVAClassifier
        factory.add_classifier('tmva', TMVAClassifier())
    except ImportError:
        pass
    factory.add_classifier('rf', RandomForestClassifier(n_estimators=10))
    factory.add_classifier('ada', AdaBoostClassifier(n_estimators=20))

    X, y, sample_weight = generate_classification_data()
    assert factory == factory.fit(X, y, sample_weight=sample_weight, features=list(X.columns), parallel_profile='threads-4')
    for cl in factory.values():
        assert list(cl.features) == list(X.columns)
    proba = factory.predict_proba(X, parallel_profile='threads-4')
    labels = factory.predict(X, parallel_profile='threads-4')
    for key, val in labels.items():
        score = accuracy_score(y, val)
        print(key, score)
        assert score > 0.7, key

    for key, val in proba.items():
        assert numpy.allclose(val.sum(axis=1), 1), 'probabilities do not sum to 1'
        assert numpy.all(val >= 0.), 'negative probabilities'

        auc_score = roc_auc_score(y, val[:, 1])
        print(auc_score)
        assert auc_score > 0.8

    for key, iterator in factory.staged_predict_proba(X).items():
        assert key != 'tmva', 'tmva does not support staged pp'
        for p in iterator:
            assert p.shape == (len(X), 2)

        # checking that last iteration coincides with previous
        assert numpy.all(p == proba[key])

    # testing picklability
    dump_string = cPickle.dumps(factory)
    clf_loaded = cPickle.loads(dump_string)

    assert type(factory) == type(clf_loaded)

    probs1 = factory.predict_proba(X)
    probs2 = clf_loaded.predict_proba(X)
    for key, val in probs1.items():
        assert numpy.all(val == probs2[key]), 'something strange was loaded'

    report = ClassificationReport({'rf': factory['rf']}, LabeledDataStorage(X, y, sample_weight))
    report.feature_importance_shuffling(roc_auc_score_mod).plot(new_plot=True, figsize=(18, 3))
    report = factory.test_on_lds(LabeledDataStorage(X, y, sample_weight))
    report = factory.test_on(X, y, sample_weight=sample_weight)
    val = numpy.mean(X['column0'])
    check_report_with_mask(report, "column0 > %f" % (val / 2.), X)
    check_report_with_mask(report, lambda x: numpy.array(x['column0']) < val * 2., X)
    check_report_with_mask(report, None, X)


def roc_auc_score_mod(y_true, prob, sample_weight=None):
    return roc_auc_score(y_true, prob[:, 1], sample_weight=sample_weight)


def check_report_with_mask(report, mask, X):
    report.roc(mask=mask).plot()
    report.prediction_pdf(mask=mask).plot()
    report.features_pdf(mask=mask).plot()
    report.efficiencies(list(X.columns), mask=mask).plot()
    report.features_correlation_matrix(mask=mask).plot()
    report.feature_importance().plot()
    report.scatter([(X.columns[0], X.columns[2])], mask=mask).plot()
    report.learning_curve(RocAuc(), mask=mask).plot()
    report.metrics_vs_cut(significance, mask=mask).plot()