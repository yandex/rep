from __future__ import division, print_function, absolute_import
from collections import OrderedDict

from sklearn import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

from rep.metaml import GridOptimalSearchCV, SubgridParameterOptimizer, FoldingScorer, \
    RegressionParameterOptimizer
from rep.test.test_estimators import generate_classification_data, check_grid, run_grid
from rep.estimators import SklearnClassifier

__author__ = 'Tatiana Likhomanenko'


def grid_tmva(score_function):
    grid_param = OrderedDict({"MaxDepth": [4, 5], "NTrees": [10, 20]})

    generator = SubgridParameterOptimizer(grid_param)
    scorer = FoldingScorer(score_function)
    from rep.estimators import TMVAClassifier

    grid = GridOptimalSearchCV(TMVAClassifier(features=['column0', 'column1']), generator, scorer)

    cl = check_grid(grid, False, False, False)
    assert 1 <= len(cl.features) <= 3
    params = cl.get_params()
    for key in grid_param:
        assert params[key] == grid.generator.best_params_[key]


def grid_sklearn(score_function):
    grid_param = OrderedDict({"n_estimators": [10, 20],
                              "learning_rate": [0.1, 0.05],
                              'features': [['column0', 'column1'], ['column0', 'column1', 'column2']]})
    generator = RegressionParameterOptimizer(grid_param)
    scorer = FoldingScorer(score_function)

    grid = GridOptimalSearchCV(SklearnClassifier(clf=AdaBoostClassifier()), generator, scorer)

    cl = check_grid(grid, False, False, False)
    assert 1 <= len(cl.features) <= 3
    params = cl.get_params()
    for key in grid_param:
        if key in params:
            assert params[key] == grid.generator.best_params_[key]
        else:
            assert params['clf__' + key] == grid.generator.best_params_[key]


def grid_custom(custom):
    grid_param = OrderedDict({"n_estimators": [10, 20],
                              "learning_rate": [0.1, 0.05],
                              'features': [['column0', 'column1'], ['column0', 'column1', 'column2']]})
    generator = SubgridParameterOptimizer(grid_param)

    grid = GridOptimalSearchCV(SklearnClassifier(clf=AdaBoostClassifier(),
                                                 features=['column0', 'column1']), generator, custom)

    cl = check_grid(grid, False, False, False)
    assert 1 <= len(cl.features) <= 3
    params = cl.get_params()
    for key in grid_param:
        if key in params:
            assert params[key] == grid.generator.best_params_[key]
        else:
            assert params['clf__' + key] == grid.generator.best_params_[key]


def test_grid():
    def generate_scorer(test, labels):
        def custom(base_estimator, params, X, y, sample_weight=None):
            cl = clone(base_estimator)
            cl.set_params(**params)
            if sample_weight is not None:
                cl.fit(X, y, sample_weight)
            else:
                cl.fit(X, y)
            return roc_auc_score(labels, cl.predict_proba(test)[:, 1])

        return custom
    X, y, _ = generate_classification_data()

    grid_custom(generate_scorer(X, y))
    run_grid(grid_sklearn)
    run_grid(grid_tmva)