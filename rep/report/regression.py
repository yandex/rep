"""
This file contains report class for regression estimators. Report includes:

    * features scatter plots, correlations
    * learning curve
    * feature importance
    * feature importance by shuffling the feature column

All methods return objects, which can have plot method (details see in :class:`rep.plotting`)
"""

from __future__ import division, print_function, absolute_import

from itertools import islice
from collections import OrderedDict
import itertools

from sklearn.metrics import mean_squared_error

from .. import plotting
from ..utils import get_columns_dict
from ._base import AbstractReport
from ..estimators.interface import Regressor

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


class RegressionReport(AbstractReport):
    def __init__(self, regressors, lds):
        """
        Report simplifies comparison of regressors on the same dataset.

        :param regressors: OrderedDict with regressors (RegressionFactory)
        :type regressors: dict[str, Regressor]
        :param LabeledDataStorage lds: data
        """
        for name, regressor in regressors.items():
            assert isinstance(regressor, Regressor), "Object {} doesn't implement interface".format(name)
        AbstractReport.__init__(self, lds=lds, estimators=regressors)

    def _predict(self, estimator, X):
        return estimator.predict(X)

    def scatter(self, correlation_pairs, mask=None, marker_size=20, alpha=0.1, grid_columns=2):
        """
        Correlation between pairs of features

        :param list[tuple] correlation_pairs: pairs of features along which scatter plot will be build.
        :param mask: mask for data, which will be used
        :type mask: None or array-like or str or function(pandas.DataFrame)
        :param int marker_size: size of marker for each event on the plot
        :param float alpha: blending parameter for scatter
        :param int grid_columns: count of columns in grid

        :rtype: plotting.GridPlot
        """
        features = list(set(itertools.chain.from_iterable(correlation_pairs)))
        _, df, = self._apply_mask(mask, self._get_features(features))
        correlation_plots = self._scatter_addition(df, correlation_pairs, marker_size=marker_size, alpha=alpha)
        return plotting.GridPlot(grid_columns, *correlation_plots)

    def predictions_scatter(self, features=None, mask=None, marker_size=20, alpha=0.1, grid_columns=2):
        """
        Correlation between predictions and features

        :param features: using features (if None then use classifier's features)
        :type features: None or list[str]
        :param mask: mask for data, which will be used
        :type mask: None or array-like or str or function(pandas.DataFrame)
        :param int marker_size: size of marker for each event on the plot
        :param float alpha: blending parameter for scatter
        :param int grid_columns: count of columns in grid

        :rtype: plotting.GridPlot
        """
        features = self.common_features if features is None else features
        mask, df, = self._apply_mask(mask, self._get_features(features))
        correlation_plots = []
        for name, prediction in self.prediction.items():
            correlation_pairs = [(feature, name) for feature in features]
            df[name] = prediction[mask]
            correlation_plots += self._scatter_addition(df, correlation_pairs, marker_size=marker_size, alpha=alpha)
        return plotting.GridPlot(grid_columns, *correlation_plots)

    @staticmethod
    def _scatter_addition(df, correlation_pairs, marker_size=20, alpha=0.1):
        correlation_plots = []
        corr_pairs = OrderedDict()
        for feature1_c, feature2_c in correlation_pairs:
            feature1, feature2 = list(get_columns_dict([feature1_c, feature2_c]).keys())
            corr_pairs[(feature1, feature2)] = (df[feature1].values, df[feature2].values)
            plot_fig = plotting.ScatterPlot({'correlation': corr_pairs[(feature1, feature2)]}, alpha=alpha,
                                            size=marker_size)
            plot_fig.xlabel = feature1
            plot_fig.ylabel = feature2
            plot_fig.figsize = (8, 6)
            correlation_plots.append(plot_fig)
        return correlation_plots

    def _learning_curve_additional(self, name, metric_func, step, mask, predict_only_masked):
        """Returns values of roc curve for particular classifier, mask and metric function. """
        evaled_mask, labels, weight = self._apply_mask(mask, self.target, self.weight)
        data = self._get_features()
        if predict_only_masked:
            _, data = self._apply_mask(mask, data)

        curve = OrderedDict()
        stage_values = self.estimators[name].staged_predict(data)
        for stage, prediction in islice(enumerate(stage_values), step - 1, None, step):
            if not predict_only_masked:
                prediction = prediction[evaled_mask]
            curve[stage] = metric_func(labels, prediction, sample_weight=weight)
        return list(curve.keys()), list(curve.values())

    def feature_importance_shuffling(self, metric=mean_squared_error, mask=None, grid_columns=2):
        """
        Get features importance using shuffling method (apply random permutation to one particular column)

        :param metric: function to measure quality
            function(y_true, y_predicted, sample_weight=None)
        :param mask: mask which points we should compare on
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param int grid_columns: number of columns in grid
        :rtype: plotting.GridPlot
        """
        return self._feature_importance_shuffling(metric=metric, mask=mask, grid_columns=grid_columns)
