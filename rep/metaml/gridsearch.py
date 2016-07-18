"""
This module does hyper parameters optimization -- finds the best parameters for estimator using different optimization models.
Components of optimization:

* estimator (for which optimal parameters are searched, any REP classifier will work, see :mod:`rep.estimators`)
* target metric function (which is maximized, anything meeting REP metric interface, see :mod:`rep.report.metrics`)
* optimization algorithm (introduced in this module)
* cross-validation technique (kFolding, introduced in this module)

During optimization, many cycles of estimating quality on different sets of parameters is done.
To speed up the process, threads or IPython cluster can be used.


GridOptimalSearchCV
-------------------
Main class linking the whole process is :class:`GridOptimalSearchCV`, which takes as parameters:

* estimator to be optimized
* scorer (which trains classifier and estimates quality using cross-validation)
* parameter generator (which draws next set of parameters to be checked)

.. autoclass:: GridOptimalSearchCV
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:


Folding Scorer
--------------
Folding cross validation can be used in grid search optimization.

.. autoclass:: ClassificationFoldingScorer
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: RegressionFoldingScorer
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:


Available optimization algorithms
---------------------------------

.. autoclass:: RandomParameterOptimizer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: AnnealingParameterOptimizer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: SubgridParameterOptimizer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: RegressionParameterOptimizer
    :members:
    :undoc-members:
    :show-inheritance:


Interface of parameter optimizer
--------------------------------
Each of parameter optimizers has the following interface.

.. autoclass:: AbstractParameterGenerator
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:


"""

from __future__ import division, print_function, absolute_import
from itertools import islice
from collections import OrderedDict
import logging
import copy

import numpy
from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.utils.random import check_random_state

from six.moves import zip
from ..estimators.utils import check_inputs
from ..utils import fit_metric
from .utils import map_on_cluster
from .utils import get_classifier_probabilities, get_regressor_prediction


__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


class AbstractParameterGenerator(object):
    def __init__(self, param_grid, n_evaluations=10, maximize=True, random_state=None):
        """
        Abstract class for grid search algorithm.
        The aim of this class is to generate new points, where the function (estimator) will be computed.
        You can define your own algorithm of step location of parameters grid.

        :param OrderedDict param_grid: the grid with parameters to optimize on
        :param int n_evaluations: the number of evaluations to do
        :param random_state: random generator
        :param maximize: whether algorithm should maximize or minimize target function.
        :type random_state: int or RandomState or None
        """
        assert isinstance(param_grid, dict), 'the passed param_grid should be of OrderedDict class'
        self.param_grid = OrderedDict(param_grid)
        _check_param_grid(param_grid)

        self.dimensions = list([len(param_values) for param, param_values in self.param_grid.items()])
        size = numpy.prod(self.dimensions)
        assert size > 1, 'The space of parameters contains only %i points' % size
        self.n_evaluations = min(n_evaluations, size)

        # results on different parameters
        self.grid_scores_ = OrderedDict()
        self.maximize = maximize

        # all the tasks that are being computed or already computed
        self.queued_tasks_ = set()
        self.random_state = check_random_state(random_state)
        self.evaluations_done = 0

    def _indices_to_parameters(self, state_indices):
        """
        Point in parameter space kept as sequence of indices, i.e.:
        max_depth: 1, 2, 4, 8
        learning_rate = 0.01, 0.1, 0.2

        Then max_depth=4, learning_rate=0.1 has internal representation as (2, 1)

        :param state_indices: sequence of integers, i.e. (1, 2)
        :return: OrderedDict, like {max_depth=4, learning_rate=0.1}
        """
        return OrderedDict([(name, values[i]) for i, (name, values) in zip(state_indices, self.param_grid.items())])

    def _generate_random_point(self, enqueue=True):
        while True:
            result = tuple([self.random_state.randint(0, size) for size in self.dimensions])
            if result not in self.queued_tasks_:
                if enqueue:
                    self.queued_tasks_.add(result)
                return result

    def generate_next_point(self):
        """Generating next random point in parameters space
        :return: tuple (indices, parameters)
        """
        raise NotImplementedError('Should be overriden by descendant')

    def generate_batch_points(self, size):
        """
        Generate several points in parameter space at once (needed when using parallel computations)

        :param size: how many points we shall generate
        :return: tuple of arrays (state_indices, state_parameters)
        """
        # may be overriden in descendants, if independent sampling is not best option.
        state_indices = []
        for _ in range(size):
            try:
                state_indices.append(self.generate_next_point())
            except StopIteration:
                pass
        return zip(*state_indices)

    def add_result(self, state_indices, value):
        """
        After the model was trained and evaluated for specific set of parameters,
        we use this function to store result
        :param state_indices: tuple, which represents the space
        :param value: quality at this point
        """
        self.grid_scores_[state_indices] = value

    def _get_the_best_value(self, value):
        if self.maximize:
            return numpy.max(value)
        else:
            return numpy.min(value)

    @property
    def best_score_(self):
        """
        Property, return best score of optimization
        """
        return self._get_the_best_value(list(self.grid_scores_.values()))

    @property
    def best_params_(self):
        """
        Property, return point of parameters grid with the best score
        """
        function = max if self.maximize else min
        return self._indices_to_parameters(function(self.grid_scores_.items(), key=lambda x: x[1])[0])

    def print_results(self, reorder=True):
        """
        Prints the results of training

        :param bool reorder: if reorder==True, best results go earlier,
         otherwise the results are printed in the order of computation
        """
        sequence = self.grid_scores_.items()
        if reorder:
            sequence = sorted(sequence, key=lambda x: x[1], reverse=self.maximize)
        for state_indices, value in sequence:
            state_string = ", ".join([name_value[0] + '=' + str(name_value[1]) for name_value
                                      in self._indices_to_parameters(state_indices).items()])
            print("{0:.3f}:  {1}".format(value, state_string))


class RandomParameterOptimizer(AbstractParameterGenerator):
    def __init__(self, param_grid, n_evaluations=10, maximize=True, random_state=None):
        """
        Works in the same way as sklearn.grid_search.RandomizedSearch.
        Each next point is generated independently.

        :param_grid: dict with distributions used to sample each parameter.
          name -> list of possible values (in which case sampled uniformly from options)
          name -> distribution (should implement '.rvs()' as scipy distributions)
        :param bool maximize: ignored parameter, added for uniformity

        NB: this is the only optimizer, which supports passing distributions for parameters.
        """
        self.maximize = maximize
        self.param_grid = OrderedDict(param_grid)
        self.n_evaluations = n_evaluations
        self.random_state = check_random_state(random_state)
        self.indices_to_parameters_ = OrderedDict()
        self.grid_scores_ = OrderedDict()
        self.queued_tasks_ = set()
        from sklearn.grid_search import ParameterSampler
        self.param_sampler = iter(ParameterSampler(param_grid, n_iter=n_evaluations, random_state=random_state))

    def generate_next_point(self):
        params = next(self.param_sampler)
        index = len(self.indices_to_parameters_)
        indices = index,
        self.indices_to_parameters_[indices] = params
        self.queued_tasks_.add(indices)
        return indices, params

    def _indices_to_parameters(self, state_indices):
        return self.indices_to_parameters_[state_indices]


class RegressionParameterOptimizer(AbstractParameterGenerator):
    def __init__(self, param_grid, n_evaluations=10, random_state=None,
                 start_evaluations=3, n_attempts=10, regressor=None, maximize=True):
        """
        This general method relies on regression.
        Regressor will try to predict the best point based on already known result fir different parameters.

        :param OrderedDict param_grid: the grid with parameters to optimize on
        :param int n_evaluations: the number of evaluations to do
        :param random_state: random generator
        :type random_state: int or RandomState or None

        :param int start_evaluations: count of random point generation on start
        :param int n_attempts: this number of points will be compared on each iteration.
            Regressor is to choose optimal from them.
        :param regressor: regressor to choose appropriate next point with potential best score
            (estimated this score by regressor); If None them RandomForest algorithm will be used.
        """
        AbstractParameterGenerator.__init__(self, param_grid=param_grid, n_evaluations=n_evaluations,
                                            random_state=random_state, maximize=maximize)
        if regressor is None:
            regressor = RandomForestRegressor(max_depth=3, n_estimators=10, max_features=0.7)
        self.regressor = regressor
        self.n_attempts = n_attempts
        self.start_evaluations = start_evaluations

    def generate_next_point(self):
        """Generating next random point in parameters space"""
        if len(self.queued_tasks_) > numpy.prod(self.dimensions) + self.n_attempts:
            raise RuntimeError("The grid is exhausted, cannot generate more points")

        if len(self.grid_scores_) < self.start_evaluations:
            new_state_indices = self._generate_random_point()
            return new_state_indices, self._indices_to_parameters(new_state_indices)

        # Training regressor
        X = numpy.array([list(x) for x in self.grid_scores_.keys()], dtype=int)
        y = list(self.grid_scores_.values())
        regressor = clone(self.regressor).fit(X, y)

        # generating candidates
        candidates = numpy.array([list(self._generate_random_point(enqueue=False))
                                  for _ in range(self.n_attempts)], dtype=int)
        # winning candidate index
        index = regressor.predict(candidates).argmax() if self.maximize else regressor.predict(candidates).argmin()

        new_state_indices = tuple(candidates[index, :])

        # remember the task
        self.queued_tasks_.add(new_state_indices)
        return new_state_indices, self._indices_to_parameters(new_state_indices)


class AnnealingParameterOptimizer(AbstractParameterGenerator):
    def __init__(self, param_grid, n_evaluations=10, temperature=0.2, random_state=None, maximize=True):
        """
        Implementation if annealing algorithm

        :param param_grid: the grid with parameters to optimize on
        :param int n_evaluations: the number od evaluations
        :param temperature: float, how tolerant we are to worse results.
            If temperature is very small, algorithm never steps to point with worse predictions.

        Doesn't support parallel execution, so cannot be used in optimization on cluster.
        """
        AbstractParameterGenerator.__init__(self, param_grid=param_grid,
                                            n_evaluations=n_evaluations,
                                            random_state=random_state, maximize=maximize)
        self.temperature = temperature
        self.actual_state = None

    def generate_next_point(self):
        """Generating next random point in parameters space"""
        if self.actual_state is None:
            new_state = self._generate_random_point(enqueue=True)
            self.actual_state = new_state
            return new_state, self._indices_to_parameters(new_state)

        else:
            actual_score = self.grid_scores_[self.actual_state]

            # checking if needed to jump after previous evaluation
            last_state = list(self.grid_scores_.keys())[-1]
            last_score = self.grid_scores_[last_state]

            # probability of transition
            std = numpy.std(list(self.grid_scores_.values())) + 1e-5
            difference = last_score - actual_score
            if not self.maximize:
                difference *= -1
            p = numpy.exp(1. / self.temperature * difference / std)
            if p > self.random_state.uniform(0, 1):
                self.actual_state = last_state

            for attempt in range(100):
                # trying to change only one of parameters
                axis = self.random_state.randint(0, len(self.dimensions))
                new_state_indices = list(self.actual_state)
                new_state_indices[axis] = self.random_state.randint(0, self.dimensions[axis])
                new_state_indices = tuple(new_state_indices)
                if new_state_indices not in self.queued_tasks_:
                    break
            else:
                print('Annealing failed to generate next point the simple way')
                new_state_indices = self._generate_random_point(enqueue=False)

            self.queued_tasks_.add(new_state_indices)
            return new_state_indices, self._indices_to_parameters(new_state_indices)

    def generate_batch_points(self, size):
        raise RuntimeError("Annealing optimization doesn't support batch-based optimization (on cluster)")


class SubgridParameterOptimizer(AbstractParameterGenerator):
    """
    Uses Metropolis-like optimization.
    If the parameter grid is large, first performs optimization on subgrid.

    :param OrderedDict param_grid: the grid with parameters to optimize on
    :param int n_evaluations: the number of evaluations to do
    :param random_state: random generator
    :type random_state: int or RandomState or None
    :param int start_evaluations: count of random point generation on start
    :param int subgrid_size: if the size of mesh too large, first we will optimize
        on subgrid with not more then subgrid_size possible values for each parameter.
    """

    def __init__(self, param_grid, n_evaluations=10, random_state=None, start_evaluations=3,
                 subgrid_size=3, maximize=True):
        AbstractParameterGenerator.__init__(self, param_grid=param_grid, n_evaluations=n_evaluations,
                                            random_state=random_state, maximize=maximize)
        self.start_evaluations = start_evaluations
        self.subgrid_size = subgrid_size
        self.dimensions_sum = sum(self.dimensions)
        self.subgrid_parameter_generator = None
        if not numpy.all(numpy.array(self.dimensions) <= 2 * self.subgrid_size):
            logger = logging.getLogger(__name__)
            logger.info("Optimizing on subgrid")
            param_subgrid, self.subgrid_indices = _create_subgrid(self.param_grid, self.subgrid_size)
            self.subgrid_parameter_generator = \
                SubgridParameterOptimizer(param_subgrid, n_evaluations=self.n_evaluations // 2,
                                          subgrid_size=subgrid_size)

    def generate_next_point(self):
        """Generating next point in parameters space"""
        if len(self.queued_tasks_) >= numpy.prod(self.dimensions):
            raise RuntimeError("The grid is exhausted, cannot generate more points")

        # trying to generate from subgrid
        if self.subgrid_parameter_generator is not None:
            if len(self.queued_tasks_) < self.subgrid_parameter_generator.n_evaluations:
                indices, parameters = self.subgrid_parameter_generator.generate_next_point()
                self.queued_tasks_.add(_translate_key_from_subgrid(self.subgrid_indices, indices))
                return ('subgrid', indices), parameters

        if len(self.grid_scores_) <= 4:
            indices = self._generate_random_point()
            self.queued_tasks_.add(indices)
            return indices, self._indices_to_parameters(indices)

        results = numpy.array(list(self.grid_scores_.values()), dtype=float)
        if not self.maximize:
            results *= -1
        std = numpy.std(results) + 1e-5
        # probabilities to take initial point
        probabilities = numpy.exp(numpy.clip((results - numpy.mean(results)) * 3. / std, -5, 5))
        probabilities /= numpy.sum(probabilities)
        # temperature is responsible for distance leaped
        temperature_p = numpy.clip(1. - len(self.queued_tasks_) / self.n_evaluations, 0.05, 1)
        while True:
            start = self.random_state.choice(len(probabilities), p=probabilities)
            start_indices = list(self.grid_scores_.keys())[start]
            new_state_indices = list(start_indices)
            for _ in range(self.dimensions_sum // 3 + 1):
                if self.random_state.uniform() < temperature_p:
                    axis = self.random_state.randint(len(self.dimensions))
                    new_state_indices[axis] += int(numpy.sign(self.random_state.uniform() - 0.5))
            if any(new_state_indices[axis] < 0 or new_state_indices[axis] >= self.dimensions[axis]
                   for axis in range(len(self.dimensions))):
                continue
            new_state_indices = tuple(new_state_indices)
            if new_state_indices in self.queued_tasks_:
                continue
            self.queued_tasks_.add(new_state_indices)
            return new_state_indices, self._indices_to_parameters(new_state_indices)

    def add_result(self, state_indices, value):
        if state_indices[0] == 'subgrid':
            self.grid_scores_[_translate_key_from_subgrid(self.subgrid_indices, state_indices[1])] = value
            self.subgrid_parameter_generator.add_result(state_indices[1], value)
        else:
            self.grid_scores_[state_indices] = value


# region supplementary functions

def _check_param_grid(param_grid):
    """ Checks parameters of grid """
    for key, v in param_grid.items():
        assert isinstance(key, str), 'Name of feature should be string'
        if isinstance(v, numpy.ndarray) and v.ndim > 1:
            raise ValueError("Parameter array should be one-dimensional.")
        if not any([isinstance(v, k) for k in (list, tuple, numpy.ndarray)]):
            raise ValueError("Parameter values should be a list.")
        if len(v) == 0:
            raise ValueError("Parameter values should be a non-empty list.")


def _create_subgrid(param_grid, n_values):
    """
    Additional function to generate subgrid

    :type param_grid: OrderedDict,
    :type n_values: int, the maximal number of values along each axis
    :rtype: (OrderedDict, OrderedDict), the subgrid and the indices of values that form subgrid
    """
    subgrid = OrderedDict()
    subgrid_indices = OrderedDict()
    for key, values in param_grid.items():
        if len(values) <= n_values:
            subgrid[key] = list(values)
            subgrid_indices[key] = range(len(values))
        else:
            # numpy.rint rounds to the nearest integer
            axis_indices = numpy.rint(numpy.linspace(-0.5, len(values) - 0.5, 2 * n_values + 1)[1::2]).astype(int)
            subgrid[key] = [values[index] for index in axis_indices]
            subgrid_indices[key] = axis_indices
    return subgrid, subgrid_indices


def _translate_key_from_subgrid(subgrid_indices, key):
    """
    :type key: tuple, the indices (describing the point) in subgrid
    :type subgrid_indices: OrderedDict, the indices of values taken to form subgrid
    :rtype: tuple, the indices in grid
    """
    return tuple([subgrid_indices[var_name][index] for var_name, index in zip(subgrid_indices, key)])


# endregion


class FoldingScorerBase(object):
    def __init__(self, score_function, folds=3, fold_checks=1, shuffle=False, random_state=None):
        """
        Scorer, which implements logic of data folding and scoring. This is a function-like object

        :param int folds: 'k' used in k-folding while validating
        :param int fold_checks: not greater than folds, the number of checks we do by cross-validating
        :param function score_function: quality. if fold_checks > 1, the average is computed over checks.

        """
        self.folds = folds
        self.fold_checks = fold_checks
        self.score_function = score_function
        self.shuffle = shuffle
        self.random_state = random_state

    def _compute_score(self, k_folder, prediction_function, base_estimator, params, X, y, sample_weight=None):
        """
        :return float: quality
        """
        score = 0
        for ind, (train_indices, test_indices) in enumerate(islice(k_folder, 0, self.fold_checks)):
            estimator = clone(base_estimator)
            estimator.set_params(**params)

            trainX, trainY = X.iloc[train_indices, :], y[train_indices]
            testX, testY = X.iloc[test_indices, :], y[test_indices]

            score_metric = copy.deepcopy(self.score_function)
            if sample_weight is not None:
                train_weights, test_weights = sample_weight[train_indices], sample_weight[test_indices]
                estimator.fit(trainX, trainY, sample_weight=train_weights)
                fit_metric(score_metric, testX, testY, sample_weight=test_weights)
                prediction = prediction_function(estimator, testX)
                score += score_metric(testY, prediction, sample_weight=test_weights)
            else:
                estimator.fit(trainX, trainY)
                fit_metric(score_metric, testX, testY)
                prediction = prediction_function(estimator, testX)
                score += score_metric(testY, prediction)
        return score / self.fold_checks


class ClassificationFoldingScorer(FoldingScorerBase):
    """
    Scorer, which implements logic of data folding and scoring for classification models. This is a function-like object

    :param int folds: 'k' used in k-folding while validating
    :param int fold_checks: not greater than folds, the number of checks we do by cross-validating
    :param function score_function: quality. if fold_checks > 1, the average is computed over checks.


    Example:

    >>> def new_score_function(y_true, proba, sample_weight=None):
    >>>     '''
    >>>     y_true: [n_samples]
    >>>     proba: [n_samples, n_classes]
    >>>     sample_weight: [n_samples] or None
    >>>     '''
    >>>     ...
    >>>
    >>> f_scorer = FoldingScorer(new_score_function)
    >>> f_scorer(base_estimator, params, X, y, sample_weight=None)
    0.5
    """
    def __call__(self, base_estimator, params, X, y, sample_weight=None):
        """
        Estimate quality of estimator with given parameters with kFolding.

        :return float: quality
        """
        k_folder = StratifiedKFold(y=y, n_folds=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        return self._compute_score(k_folder, get_classifier_probabilities, base_estimator, params, X, y,
                                   sample_weight=sample_weight)


class RegressionFoldingScorer(FoldingScorerBase):
    """
    Scorer, which implements logic of data folding and scoring for regression models. This is a function-like object

    :param int folds: 'k' used in k-folding while validating
    :param int fold_checks: not greater than folds, the number of checks we do by cross-validating
    :param function score_function: quality. if fold_checks > 1, the average is computed over checks.


    Example:

    >>> def new_score_function(y_true, pred, sample_weight=None):
    >>>     '''
    >>>     y_true: [n_samples]
    >>>     pred: [n_samples]
    >>>     sample_weight: [n_samples] or None
    >>>     '''
    >>>     ...
    >>>
    >>> f_scorer = RegressionFoldingScorer(new_score_function)
    >>> f_scorer(base_estimator, params, X, y, sample_weight=None)
    0.5
    """
    def __call__(self, base_estimator, params, X, y, sample_weight=None):
        """
        Estimate quality of estimator with given parameters with kFolding.

        :return float: quality
        """
        k_folder = KFold(len(y), n_folds=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        return self._compute_score(k_folder, get_regressor_prediction, base_estimator, params, X, y,
                                   sample_weight=sample_weight)

FoldingScorer = ClassificationFoldingScorer


def apply_scorer(scorer, params, base_estimator, X, y, sample_weight):
    """
    Application of scorer algorithm.

    :param scorer: algorithm to train estimator and get quality (see FoldingScorer for example)
    :param dict params: parameters for estimator
    :param base.BaseEstimator base_estimator: estimator
    :param X: pandas.DataFrame of shape [n_samples, n_features], data
    :param y: labels of events - array-like of shape [n_samples]
    :param sample_weight: weight of events,
           array-like of shape [n_samples] or None if all weights are equal

    :return: ('success', float) or ('fail', Exception), float will contain result.
    """
    try:
        estimator = clone(base_estimator)
        return 'success', scorer(params=params, base_estimator=estimator, X=X, y=y, sample_weight=sample_weight)
    except Exception as e:
        return 'fail', e


class GridOptimalSearchCV(object):
    """
    Optimal search over specified parameter values for an estimator.
    Uses different optimization techniques to use limited number of evaluations without using exhaustive grid scanning.

    GridSearchCV implements a "fit" method and a "fit_best_estimator" method to train models.

    :param BaseEstimator estimator: object of type that implements the "fit" and "fit_best_estimator" methods
        A new object of that type is cloned for each point.
    :param AbstractParameterGenerator params_generator: generator of grid search algorithm
    :param object scorer: which implement method __call__ with kwargs:
        "base_estimator", "params", "X", "y", "sample_weight"
    :param parallel_profile: name of profile
    :type parallel_profile: None or str

    **Attributes**:

    generator: return grid parameter generator
    """\

    def __init__(self, estimator, params_generator, scorer, parallel_profile=None):
        self.base_estimator = estimator
        self.params_generator = params_generator
        self.scorer = scorer
        self.parallel_profile = parallel_profile
        self.evaluations_done = 0

    @staticmethod
    def _log(msg, level=20):
        logger = logging.getLogger(__name__)
        logger.log(level, msg)

    @property
    def generator(self):
        """Property for params_generator"""
        return self.params_generator

    def fit_best_estimator(self, X, y, sample_weight=None):
        """
        Train estimator with the best parameters

        :param X: pandas.DataFrame of shape [n_samples, n_features]
        :param y: labels of events - array-like of shape [n_samples]
        :param sample_weight: weight of events,
               array-like of shape [n_samples] or None if all weights are equal

        :return: the best estimator
        """
        best_estimator_ = clone(self.base_estimator)
        best_estimator_.set_params(**self.generator.best_params_)
        best_estimator_.fit(X, y, sample_weight=sample_weight)
        return best_estimator_

    def fit(self, X, y, sample_weight=None):
        """
        Run fit with all sets of parameters.

        :param X: array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        :param y: array-like, shape = [n_samples] or [n_samples, n_output], optional
        :param sample_weight: array-like, shape = [n_samples], weight
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=True)

        if self.parallel_profile is None:
            while self.evaluations_done < self.params_generator.n_evaluations:
                state_indices, state_dict = self.params_generator.generate_next_point()
                status, value = apply_scorer(self.scorer, state_dict, self.base_estimator, X, y, sample_weight)
                assert status == 'success', 'Error during grid search ' + str(value)
                self.params_generator.add_result(state_indices, value)
                self.evaluations_done += 1
                state_string = ", ".join([k + '=' + str(v) for k, v in state_dict.items()])
                self._log('{}: {}'.format(value, state_string))
        else:
            if str.startswith(self.parallel_profile, 'threads'):
                _, n_threads = str.split(self.parallel_profile, '-')
                portion = int(n_threads)
                print("Performing grid search in {} threads".format(portion))
            else:
                from IPython.parallel import Client

                direct_view = Client(profile=self.parallel_profile).direct_view()
                portion = len(direct_view)
                print("There are {0} cores in cluster, the portion is equal {1}".format(len(direct_view), portion))

            while self.evaluations_done < self.params_generator.n_evaluations:
                state_indices_array, state_dict_array = self.params_generator.generate_batch_points(size=portion)
                current_portion = len(state_indices_array)
                result = map_on_cluster(self.parallel_profile, apply_scorer, [self.scorer] * current_portion,
                                        state_dict_array,
                                        [self.base_estimator] * current_portion,
                                        [X] * current_portion, [y] * current_portion, [sample_weight] * current_portion)
                assert len(result) == current_portion, "The length of result is very strange"
                for state_indices, state_dict, (status, score) in zip(state_indices_array, state_dict_array, result):
                    params = ", ".join([k + '=' + str(v) for k, v in state_dict.items()])
                    if status != 'success':
                        message = 'Fail during training on the node \nException {exc}\n Parameters {params}'
                        self._log(message.format(exc=score, params=params), level=40)
                    else:
                        self.params_generator.add_result(state_indices, score)
                        self._log("{}: {}".format(score, params))
                self.evaluations_done += current_portion
                print("%i evaluations done" % self.evaluations_done)
        return self
