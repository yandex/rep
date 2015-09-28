from __future__ import division, print_function, absolute_import

import numpy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, log_loss

from rep.data.storage import LabeledDataStorage
from rep.metaml import ClassifiersFactory, RegressorsFactory
from rep.test.test_estimators import generate_classification_sample, generate_regression_sample
from rep.report.metrics import RocAuc, LogLoss

__author__ = 'Alex Rogozhnikov'


# Important:
# unfortunately, testing is very complicated, so only workability is tested


def check_classification_learning_curve_masks(report, n_classes):
    """testing predict_only_mask option"""
    for mask in [None, "column0 > 0", "column0 > -0.5"]:
        if n_classes == 2:
            loss = RocAuc()
        else:
            loss = log_loss
        lc1 = report.learning_curve(loss, mask=mask, predict_only_masked=True, steps=3)
        lc2 = report.learning_curve(loss, mask=mask, predict_only_masked=False, steps=3)
        assert lc1.functions == lc2.functions


def check_regression_learning_curve_masks(report):
    """testing predict_only_mask option"""
    for mask in [None, "column0 > 0", "column0 > -0.5"]:
        loss = mean_squared_error
        lc1 = report.learning_curve(loss, mask=mask, predict_only_masked=True, steps=3)
        lc2 = report.learning_curve(loss, mask=mask, predict_only_masked=False, steps=3)
        assert lc1.functions == lc2.functions


def _classification_mask_report(report, mask, X, labels_dict):
    report.features_correlation_matrix(mask=mask).plot()
    report.features_correlation_matrix_by_class(mask=mask, labels_dict=labels_dict).plot()
    report.efficiencies(features=X.columns[1:3], mask=mask, labels_dict=labels_dict).plot()
    report.features_pdf(mask=mask, labels_dict=labels_dict).plot()
    report.learning_curve(RocAuc(), mask=mask, metric_label='roc').plot()
    significance = lambda s, b: s / (numpy.sqrt(b) + 0.01)
    report.metrics_vs_cut(significance, mask=mask, metric_label='sign').plot()
    report.prediction_pdf(mask=mask, labels_dict=labels_dict).plot()
    report.scatter([X.columns[:2], X.columns[1:3]], mask=mask, labels_dict=labels_dict).plot()
    report.feature_importance().plot()
    if labels_dict is None:
        report.feature_importance_shuffling(mask=mask).plot()
        report.roc(mask=mask).plot()
    report.efficiencies_2d(['column0', 'column1'], 0.3, mask=mask, labels_dict=labels_dict)
    print(report.compute_metric(RocAuc()))


def _check_classification_report(n_classes=2):
    classifiers = ClassifiersFactory()
    classifiers.add_classifier('gb', GradientBoostingClassifier(n_estimators=10))
    classifiers.add_classifier('rf', RandomForestClassifier())
    classifiers.add_classifier('ada', AdaBoostClassifier(n_estimators=10))

    X, y = generate_classification_sample(1000, 5, n_classes=n_classes)
    classifiers.fit(X, y)

    X, y = generate_classification_sample(1000, 5, n_classes=n_classes)
    test_lds = LabeledDataStorage(X, y, sample_weight=None)
    report = classifiers.test_on_lds(test_lds)

    val = numpy.mean(X['column0'])
    labels_dict = None
    if n_classes > 2:
        labels_dict = {i: str(i) for i in range(n_classes)}
    _classification_mask_report(report, "column0 > %f" % val, X, labels_dict)
    _classification_mask_report(report, lambda x: numpy.array(x['column0']) < val, X, labels_dict)
    _classification_mask_report(report, None, X, labels_dict)
    check_classification_learning_curve_masks(report, n_classes=n_classes)


def test_classification_report():
    _check_classification_report(n_classes=2)


def test_multiclassification_report():
    _check_classification_report(n_classes=4)


def test_regression_report():
    regressors = RegressorsFactory()
    regressors.add_regressor('gb', GradientBoostingRegressor(n_estimators=10))
    regressors.add_regressor('rf', RandomForestRegressor())
    regressors.add_regressor('ada', AdaBoostRegressor(n_estimators=10))

    X, y = generate_regression_sample(1000, 5)
    regressors.fit(X, y)

    X, y = generate_regression_sample(1000, 5)
    test_lds = LabeledDataStorage(X, y, sample_weight=None)
    regression_report = regressors.test_on_lds(test_lds)
    val = numpy.mean(X['column0'])
    _regression_mask_report(regression_report, "column0 > %f" % val, X)
    _regression_mask_report(regression_report, lambda x: numpy.array(x['column0']) < val, X)
    _regression_mask_report(regression_report, None, X)
    check_regression_learning_curve_masks(regression_report)


def _regression_mask_report(report, mask, X):
    report.features_correlation_matrix(mask=mask).plot()
    report.learning_curve(mean_squared_error, mask=mask, metric_label='mse').plot()

    report.scatter([X.columns[:2], X.columns[1:3]], mask=mask).plot()
    report.predictions_scatter(mask=mask).plot()
    report.features_correlation_matrix(mask=mask).plot()
    report.feature_importance().plot()
    report.feature_importance_shuffling(mask=mask).plot()
    print(report.compute_metric(mean_squared_error))
