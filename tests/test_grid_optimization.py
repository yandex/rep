from __future__ import division, print_function, absolute_import
from collections import OrderedDict

from rep.metaml.gridsearch import SubgridParameterOptimizer, \
    RegressionParameterOptimizer, AbstractParameterGenerator, \
    AnnealingParameterOptimizer, RandomParameterOptimizer
import numpy
from tests import retry_if_fails

__author__ = 'Alex Rogozhnikov'


class FunctionOptimizer(object):
    """Class was created to test different optimizing algorithms on functions,
    it gets any function of several variables and just optimizes it"""

    def __init__(self, function, param_grid, maximize, n_evaluations=100, parameter_generator_type=None):
        """
        :type function: some function, we are looking for its maximal value.
        :type parameter_generator_type: (param_grid, n_evaluations) -> AbstractParameterGenerator
        """
        self.function = function
        self.generator = parameter_generator_type(param_grid=param_grid, n_evaluations=n_evaluations, maximize=maximize)

    def optimize(self):
        assert isinstance(self.generator, AbstractParameterGenerator), 'the generator should be an instance of ' \
                                                                       'abstract parameter generator'
        for _ in range(self.generator.n_evaluations):
            next_indices, next_params = self.generator.generate_next_point()
            value = self.function(**next_params)
            self.generator.add_result(state_indices=next_indices, value=value)

    def print_results(self, reorder=True):
        self.generator.print_results(reorder=reorder)


@retry_if_fails
def test_parameter_generators(n_evaluations=100):
    """Testing optimizers on a basic problem of function optimization """
    for generator_type in [RandomParameterOptimizer,
                           RegressionParameterOptimizer,
                           SubgridParameterOptimizer,
                           AnnealingParameterOptimizer,
                           ]:
        for maximize in [True, False]:
            yield check_optimizer, generator_type, maximize, n_evaluations


def check_optimizer(generator_type, maximize, n_evaluations):
    parameters = {
        'x': numpy.linspace(0.1, 1, 10),
        'y': numpy.linspace(0.1, 1, 10),
        'z': numpy.linspace(0.1, 1, 10),
        'w': numpy.linspace(0.1, 1, 10),
    }
    parameters = OrderedDict(parameters)
    print(generator_type.__name__, 'maximize', maximize)
    sign = 2 * maximize - 1
    optimizer = FunctionOptimizer(lambda x, y, z, w: sign * x * y * z * w,
                                  parameter_generator_type=generator_type,
                                  maximize=maximize,
                                  param_grid=parameters,
                                  n_evaluations=n_evaluations)
    optimizer.optimize()
    assert len(optimizer.generator.grid_scores_) == n_evaluations
    assert len(optimizer.generator.queued_tasks_) == n_evaluations
    assert set(optimizer.generator.grid_scores_.keys()) == optimizer.generator.queued_tasks_
    scores = list(optimizer.generator.grid_scores_.values())
    expected_mean = numpy.prod([numpy.mean(p) for p in parameters.values()]) * sign
    if generator_type is not RandomParameterOptimizer:
        # check that quality is better than random
        passed_check = (numpy.mean(scores) - expected_mean) * sign > 0
        if not passed_check:
            optimizer.print_results()
            print('\n\n')
            optimizer.print_results(reorder=False)

        assert passed_check, \
            "Generator {}, maximize {}, computed mean {}, expected_mean {}".format(
                generator_type.__name__, maximize, numpy.mean(scores), expected_mean)


def test_random_optimization_with_distributions(n_evaluations=60):
    from scipy.stats import norm, expon
    parameters = OrderedDict()
    parameters['x'] = numpy.linspace(0.1, 1, 10)
    parameters['y'] = norm(0, 1)
    parameters['z'] = expon(3)
    parameters['w'] = numpy.linspace(0.1, 1, 10)

    optimizer = FunctionOptimizer(lambda x, y, z, w: x * y * z * w,
                                  parameter_generator_type=RandomParameterOptimizer,
                                  maximize=True,
                                  param_grid=parameters,
                                  n_evaluations=n_evaluations)
    optimizer.optimize()
    assert len(optimizer.generator.grid_scores_) == n_evaluations
    assert len(optimizer.generator.queued_tasks_) == n_evaluations
    assert set(optimizer.generator.grid_scores_.keys()) == optimizer.generator.queued_tasks_
    optimizer.print_results()
