from __future__ import division, print_function, absolute_import

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics.metrics import mean_squared_error
import numpy

from rep.data import LabeledDataStorage
from rep.metaml import RegressorsFactory
from six.moves import cPickle
from rep.report import RegressionReport
from rep.test.test_estimators import generate_classification_data


__author__ = 'Tatiana Likhomanenko'

# TODO testing of right-classification of estimators


def test_factory():
    factory = RegressorsFactory()
    try:
        from rep.estimators.tmva import TMVARegressor
        factory.add_regressor('tmva', TMVARegressor())
    except ImportError:
        pass
    factory.add_regressor('rf', RandomForestRegressor(n_estimators=10))
    factory.add_regressor('ada', AdaBoostRegressor(n_estimators=20))

    X, y, sample_weight = generate_classification_data()
    assert factory == factory.fit(X, y, sample_weight=sample_weight, features=list(X.columns))
    values = factory.predict(X)

    for cl in factory.values():
        assert list(cl.features) == list(X.columns)

    for key, val in values.items():
        score = mean_squared_error(y, val)
        print(score)
        assert score < 0.2

    for key, iterator in factory.staged_predict(X).items():
        assert key != 'tmva', 'tmva does not support staged pp'
        for p in iterator:
            assert p.shape == (len(X), )

        # checking that last iteration coincides with previous
        assert numpy.all(p == values[key])

    # testing picklability
    dump_string = cPickle.dumps(factory)
    clf_loaded = cPickle.loads(dump_string)

    assert type(factory) == type(clf_loaded)

    probs1 = factory.predict(X)
    probs2 = clf_loaded.predict(X)
    for key, val in probs1.items():
        assert numpy.all(val == probs2[key]), 'something strange was loaded'

    report = RegressionReport({'rf': factory['rf']}, LabeledDataStorage(X, y, sample_weight))
    report.feature_importance_shuffling(mean_squared_mod).plot(new_plot=True, figsize=(18, 3))
    report = factory.test_on_lds(LabeledDataStorage(X, y, sample_weight))
    report = factory.test_on(X, y, sample_weight=sample_weight)
    report.feature_importance()
    report.features_correlation_matrix()
    report.predictions_scatter()

    val = numpy.mean(X['column0'])
    report_mask(report, "column0 > %f" % val, X)
    report_mask(report, lambda x: numpy.array(x['column0']) < val, X)
    report_mask(report, None, X)


def mean_squared_mod(y_true, values, sample_weight=None):
    return mean_squared_error(y_true, values, sample_weight=sample_weight)


def report_mask(report, mask, X):
    report.features_correlation_matrix(mask=mask).plot()
    report.feature_importance().plot()
    report.scatter([(X.columns[0], X.columns[2])], mask=mask).plot()
    report.predictions_scatter([X.columns[0], X.columns[2]], mask=mask).plot()
    report.learning_curve(mean_squared_error, mask=mask).plot()
