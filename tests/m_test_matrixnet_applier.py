"""
Here we test the correctness and speed of the formula.
"""

from __future__ import division, print_function, absolute_import
import os
import time
import numpy
from scipy.special import expit
import pandas
from six import BytesIO
from six.moves import zip
from rep.estimators._matrixnetapplier import MatrixNetApplier as NumpyClassifier


__author__ = 'Alex Rogozhnikov'

DATA_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "help_files")


def read_files(mx_filename, test_filename):
    test_file = pandas.read_csv(test_filename, sep='\t')
    with open(mx_filename, 'rb') as mx:
        mx_content = mx.read()
    return mx_content, test_file


def numpy_predict(formula_mx, data):
    data = data.astype(float)
    data = pandas.DataFrame(data)
    mx = NumpyClassifier(BytesIO(formula_mx))
    return mx.apply(data)


def stage_numpy_predict(formula_mx, data, step=1):
    data = data.astype(float)
    data = pandas.DataFrame(data)
    mx = NumpyClassifier(BytesIO(formula_mx))

    prediction = numpy.zeros(len(data))

    for num, prediction_iteration in enumerate(mx.apply_separately(data)):
        prediction += prediction_iteration
        if num % step == 0:
            yield expit(prediction)


def check_leaves(mx_filename, test_filename, n_trees=5000):
    formula_mx, data = read_files(mx_filename, test_filename)
    data = data.astype(float)
    data = pandas.DataFrame(data)
    mx = NumpyClassifier(BytesIO(formula_mx))
    leaves = mx.compute_leaf_indices(data)
    assert leaves.shape[0] == data.shape[0]
    assert leaves.shape[1] == n_trees
    print(leaves)


def test_leaves():
    check_leaves(
            os.path.join(DATA_PATH, 'test_formula_mx'),
            os.path.join(DATA_PATH, 'data.csv'))


def check_staged_predictions(mx_filename, test_filename, n_iterations, stage_predict_function):
    mx_content, test_file = read_files(mx_filename, test_filename)

    predictions = pandas.read_csv(os.path.join(DATA_PATH, 'predictions.csv'))
    predictions = pandas.DataFrame(predictions)

    # Checking the predictions on first 100 events
    for x, (key, row) in zip(stage_predict_function(mx_content, test_file[:100]), predictions.iterrows()):
        assert numpy.allclose(row, x)

    # Checking the number of iterations on 10 events
    assert sum(1 for _ in stage_predict_function(mx_content, test_file[:10])) == n_iterations + 1

    print('Check was passed')


# How the file was obtained
# def write_staged_predictions(mx_filename, test_filename):
#   mx_content, test_file = read_files(mx_filename, test_filename)
#   # testing on first 100 events
#   test_file = test_file[:100]
#
#   predictions = numpy.zeros([100, 100], dtype=float)
#
# for i, x in enumerate(stage_cython_predict(mx_content, test_file)):
#   if i == 100:
#       break
#   predictions[i, :] = x
#
# pandas.DataFrame(predictions).to_csv('data/predictions.csv', index=False)


def compute_speed(mx_filename, test_filename, function, print_name=''):
    mx_content, test_file = read_files(mx_filename, test_filename)
    # just iterating over sequence
    start = time.time()
    for x in function(mx_content, test_file):
        pass
    print(print_name, time.time() - start)


def test_applier():
    check_staged_predictions(
            os.path.join(DATA_PATH, 'test_formula_mx'),
            os.path.join(DATA_PATH, 'data.csv'),
            stage_predict_function=stage_numpy_predict,
            n_iterations=5000)