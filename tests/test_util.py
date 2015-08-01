from __future__ import print_function, division, absolute_import

__author__ = 'Tatiana Likhomanenko'

from rep import utils
import numpy
import pandas


def test_calc():
    prediction = numpy.random.random(10000)
    iron = utils.Flattener(prediction)
    assert numpy.allclose(numpy.histogram(iron(prediction), normed=True, bins=30)[0], numpy.ones(30), rtol=1e-02)

    x, y, yerr, xerr = utils.calc_hist_with_errors(iron(prediction), bins=30, x_range=(0, 1))
    assert numpy.allclose(y, numpy.ones(len(y)), rtol=1e-02)
    width = 1. / 60
    means = numpy.linspace(width, 1 - width, 30)
    assert numpy.allclose(x, means)
    assert numpy.allclose(xerr, numpy.zeros(len(xerr)) + width)
    assert numpy.allclose(yerr, numpy.zeros(len(yerr)) + yerr[0], rtol=1e-2)

    random_labels = numpy.random.choice(2, size=10000)
    (tpr, tnr), _, _ = utils.calc_ROC(prediction, random_labels)

    # checking for random classifier
    assert numpy.max(abs(1 - tpr - tnr)) < 0.05

    # checking efficiencies for random mass, random prediction
    mass = numpy.random.random(10000)
    result = utils.get_efficiencies(prediction, mass)
    for threshold, (xval, yval) in result.items():
        assert ((yval + threshold - 1) ** 2).mean() < 0.1


def test_train_test_split_group():
    data = list(range(50)) * 2
    group_column = list(range(50)) * 2
    train, test = utils.train_test_split_group(group_column, data)
    assert len(set.intersection(set(test), set(train))) == 0


def test_corr_coeff(n_samples=1000):
    weights = numpy.ones(n_samples)
    df = pandas.DataFrame(data=numpy.random.random([n_samples, 10]))
    z1 = numpy.corrcoef(df.values.T)
    z2 = utils.calc_feature_correlation_matrix(df)
    z3 = utils.calc_feature_correlation_matrix(df, weights=weights)
    assert numpy.allclose(z1, z2)
    assert numpy.allclose(z1, z3)
