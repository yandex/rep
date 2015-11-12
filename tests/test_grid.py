from __future__ import division, print_function, absolute_import
from collections import OrderedDict

from sklearn import clone
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.qda import QDA

from rep.test.test_estimators import check_classification_model, check_regression_model

from rep.metaml import GridOptimalSearchCV, SubgridParameterOptimizer, FoldingScorer, \
    RegressionParameterOptimizer, RegressionFoldingScorer, ClassificationFoldingScorer
from rep.metaml.gridsearch import AnnealingParameterOptimizer, RandomParameterOptimizer
from rep.report.metrics import OptimalAMS, RocAuc, LogLoss, OptimalSignificance
from rep.test.test_estimators import generate_classification_data, generate_regression_data
from rep.estimators import SklearnClassifier, SklearnRegressor
from sklearn.metrics import mean_squared_error
import numpy

__author__ = 'Tatiana Likhomanenko'


def check_grid(estimator, check_instance=True, has_staged_pp=True, has_importances=True, use_weights=False,
               classification=True):
    if classification:
        X, y, sample_weight = generate_classification_data()
    else:
        X, y, sample_weight = generate_regression_data()
    assert len(sample_weight) == len(X), 'somehow lengths are different'

    if use_weights:
        assert estimator == estimator.fit(X, y, sample_weight=sample_weight)
        estimator = estimator.fit_best_estimator(X, y, sample_weight=sample_weight)
    else:
        assert estimator == estimator.fit(X, y)
        estimator = estimator.fit_best_estimator(X, y)

    if classification:
        check_classification_model(estimator, X, y, check_instance=check_instance, has_staged_pp=has_staged_pp,
                                   has_importances=has_importances)
    else:
        check_regression_model(estimator, X, y, check_instance=check_instance, has_stages=has_staged_pp,
                               has_importances=has_importances)

    return estimator


def test_gridsearch_on_tmva():
    metric = numpy.random.choice([OptimalAMS(), RocAuc()])
    scorer = FoldingScorer(metric)

    grid_param = OrderedDict({"MaxDepth": [4, 5], "NTrees": [10, 20]})
    generator = SubgridParameterOptimizer(n_evaluations=5, param_grid=grid_param)

    try:
        from rep.estimators import TMVAClassifier

        base_tmva = TMVAClassifier(features=['column0', 'column1'], method='kBDT')
        grid = GridOptimalSearchCV(base_tmva, generator, scorer)
        classifier = check_grid(grid, False, False, False)
        # checking parameters
        assert len(classifier.features) == 2
        params = classifier.get_params()
        for key in grid_param:
            assert params[key] == grid.generator.best_params_[key]
    except ImportError:
        pass


def test_gridsearch_sklearn():
    metric = numpy.random.choice([OptimalAMS(), RocAuc(), LogLoss()])
    scorer = ClassificationFoldingScorer(metric)
    maximization = True
    if isinstance(metric, LogLoss):
        maximization = False
    grid_param = OrderedDict({"n_estimators": [10, 20],
                              "learning_rate": [0.1, 0.05],
                              'features': [['column0', 'column1'], ['column0', 'column1', 'column2']]})
    generator = RegressionParameterOptimizer(grid_param, n_evaluations=4, maximize=maximization)

    grid = GridOptimalSearchCV(SklearnClassifier(clf=AdaBoostClassifier()), generator, scorer,
                               parallel_profile='threads-3')

    _ = check_grid(grid, False, False, False, use_weights=True)
    classifier = check_grid(grid, False, False, False, use_weights=False)

    # Check parameters of best fitted classifier
    assert 2 <= len(classifier.features) <= 3, 'Features were not set'
    params = classifier.get_params()
    for key in grid_param:
        if key in params:
            assert params[key] == grid.generator.best_params_[key]
        else:
            assert params['clf__' + key] == grid.generator.best_params_[key]


def test_gridsearch_sklearn_regression():
    scorer = RegressionFoldingScorer(mean_squared_error)

    grid_param = OrderedDict({"n_estimators": [10, 20],
                              "learning_rate": [0.1, 0.05],
                              'features': [['column0', 'column1'], ['column0', 'column1', 'column2']]})
    generator = RegressionParameterOptimizer(grid_param, n_evaluations=4)

    grid = GridOptimalSearchCV(SklearnRegressor(clf=AdaBoostRegressor()), generator, scorer)
    # parallel_profile='threads-3')

    _ = check_grid(grid, False, False, False, use_weights=True, classification=False)
    regressor = check_grid(grid, False, False, False, use_weights=False, classification=False)

    # Check parameters of best fitted classifier
    assert 2 <= len(regressor.features) <= 3, 'Features were not set'
    params = regressor.get_params()
    for key in grid_param:
        if key in params:
            assert params[key] == grid.generator.best_params_[key]
        else:
            assert params['clf__' + key] == grid.generator.best_params_[key]


def test_gridsearch_threads(n_threads=3):
    scorer = FoldingScorer(numpy.random.choice([OptimalAMS(), RocAuc()]))

    grid_param = OrderedDict({"n_estimators": [10, 20],
                              "learning_rate": [0.1, 0.05],
                              'features': [['column0', 'column1'], ['column0', 'column1', 'column2']]})
    generator = RegressionParameterOptimizer(grid_param, n_evaluations=4)

    base = SklearnClassifier(clf=AdaBoostClassifier())
    grid = GridOptimalSearchCV(base, generator, scorer, parallel_profile='threads-{}'.format(n_threads))

    X, y, sample_weight = generate_classification_data()
    grid.fit(X, y, sample_weight=sample_weight)


def test_grid_with_custom_scorer():
    """
    Introducing here special scorer which always uses all data passed to gridsearch.fit as training
    and tests on another fixed dataset (which was passed to scorer) bu computing roc_auc_score from sklearn.
    """

    class CustomScorer(object):
        def __init__(self, testX, testY):
            self.testY = testY
            self.testX = testX

        def __call__(self, base_estimator, params, X, y, sample_weight=None):
            cl = clone(base_estimator)
            cl.set_params(**params)
            if sample_weight is not None:
                cl.fit(X, y, sample_weight)
            else:
                cl.fit(X, y)
            return roc_auc_score(self.testY, cl.predict_proba(self.testX)[:, 1])

    X, y, _ = generate_classification_data()
    custom_scorer = CustomScorer(X, y)

    grid_param = OrderedDict({"n_estimators": [10, 20],
                              "learning_rate": [0.1, 0.05],
                              'features': [['column0', 'column1'], ['column0', 'column1', 'column2']]})
    generator = SubgridParameterOptimizer(grid_param)

    base_estimator = SklearnClassifier(clf=AdaBoostClassifier())
    grid = GridOptimalSearchCV(base_estimator, generator, custom_scorer)

    cl = check_grid(grid, False, False, False)
    assert len(cl.features) <= 3
    params = cl.get_params()
    for key in grid_param:
        if key in params:
            assert params[key] == grid.generator.best_params_[key]
        else:
            assert params['clf__' + key] == grid.generator.best_params_[key]


def test_gridsearch_metrics():
    X, y, sample_weight = generate_classification_data(n_classes=2, distance=0.7)
    param_grid = OrderedDict({
        'reg_param': numpy.linspace(0, 1, 20)
    })

    from itertools import cycle

    optimizers = cycle([
        RegressionParameterOptimizer(param_grid=param_grid, n_evaluations=4, start_evaluations=2),
        AnnealingParameterOptimizer(param_grid=param_grid, n_evaluations=4),
        SubgridParameterOptimizer(param_grid=param_grid, n_evaluations=4),
        RandomParameterOptimizer(param_grid=param_grid, n_evaluations=4),
    ])

    for metric in [RocAuc(), OptimalAMS(), OptimalSignificance(), log_loss]:
        scorer = FoldingScorer(metric)
        clf = SklearnClassifier(QDA())
        grid = GridOptimalSearchCV(estimator=clf, params_generator=next(optimizers),
                                   scorer=scorer)
        grid.fit(X, y)
        print(grid.params_generator.best_score_)
        print(grid.params_generator.best_params_)
        grid.params_generator.print_results()


def test_gridsearch_metrics_threads(n_threads=3):
    X, y, sample_weight = generate_classification_data(n_classes=2, distance=0.7)
    param_grid = OrderedDict({
        'reg_param': numpy.linspace(0, 1, 20)
    })

    from itertools import cycle

    optimizers = cycle([
        RegressionParameterOptimizer(param_grid=param_grid, n_evaluations=4, start_evaluations=2),
        SubgridParameterOptimizer(param_grid=param_grid, n_evaluations=4),
        RandomParameterOptimizer(param_grid=param_grid, n_evaluations=4),
    ])

    for metric in [RocAuc(), OptimalAMS(), OptimalSignificance(), log_loss]:
        scorer = FoldingScorer(metric)
        clf = SklearnClassifier(QDA())
        grid = GridOptimalSearchCV(estimator=clf, params_generator=next(optimizers),
                                   scorer=scorer, parallel_profile='threads-{}'.format(n_threads))
        grid.fit(X, y)
        print(grid.params_generator.best_score_)
        print(grid.params_generator.best_params_)
        grid.params_generator.print_results()
