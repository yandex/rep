from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import numpy
import pandas
import copy
from collections import OrderedDict
from .. import plotting
from .. import utils

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


class AbstractReport:
    """
    Provides methods used both in Classification and Regression reports

    Parameters:
    -----------
    :type lds: rep.data.storage.LabeledDataStorage
    :type estimators: dict[str, Classifier] or dict[str, Regressor]
    """
    __metaclass__ = ABCMeta

    def __init__(self, estimators, lds):
        self.lds = lds
        if not isinstance(estimators, OrderedDict):
            estimators = OrderedDict(estimators)

        self.estimators = estimators

        self.prediction = OrderedDict()
        X = lds.get_data()
        for name, estimator in self.estimators.items():
            self.prediction[name] = self._predict(estimator, X)

        self.target, self.weight = lds.get_targets(), lds.get_weights()

        self.common_features = list(
            set.intersection(*[set(estimator.features) for name, estimator in self.estimators.items()]))

    @abstractmethod
    def _predict(self, estimator, X):
        """Returns probabilities for estimators and predictions for regressors"""
        pass

    def _apply_mask(self, mask, *args):
        if mask is None:
            return tuple([numpy.ones(len(self.lds), dtype=bool)] + list(args))
        mask = self.lds.eval_column(mask)
        mask_data = [data.iloc[mask, :] if isinstance(data, pandas.DataFrame) else data[mask] for data in args]
        return tuple([mask] + mask_data)

    def _get_features(self, features=None):
        return self.lds.get_data(features=features)

    def features_correlation_matrix(self, features=None, mask=None, tick_labels=None, vmin=-1, vmax=1, cmap='Reds'):
        """
        Correlation between features

        :param features: using features (if None then use estimator's features)
        :type features: None or list[str]
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param tick_labels: names for features in matrix
        :type tick_labels: None or array-like
        :param int vmin: min of value for min color
        :param int vmax: max of value for max color
        :param str cmap: color map name
        :rtype: plotting.ColorMap
        """
        features = self.common_features if features is None else features
        _, df, = self._apply_mask(mask, self._get_features(features))
        features_names = list(df.columns)
        if tick_labels is None:
            tick_labels = features_names

        assert len(tick_labels) == len(features_names), 'Tick labels and features have different length'
        plot_corr = plotting.ColorMap(
            utils.calc_feature_correlation_matrix(df[features_names]),
            labels=tick_labels, vmin=vmin, vmax=vmax, cmap=cmap)
        plot_corr.title = 'Correlation'
        plot_corr.fontsize = 10
        plot_corr.figsize = (len(features) // 5 + 2, len(features) // 5)

        return plot_corr

    def learning_curve(self, metric, mask=None, steps=10, metric_label='metric', predict_only_masked=True):
        """
        Get learning curves

        :param function metric: function looks like function
            def function(y_true, y_pred, sample_weight=None)
        :param steps: if int, the same step is used in all learning curves,
            otherwise dict with steps for each estimator
        :type steps: int or dict
        :param str metric_label: name for metric on plot
        :param bool predict_only_masked: if True, will predict only for needed events.
          When you build learning curves for FoldingClassifier/FoldingRegressor on the same dataset,
          set this to False to get unbiased predictions.

        :rtype: plotting.FunctionsPlot
        """
        mask, data, labels, weight = self._apply_mask(mask, self._get_features(), self.target, self.weight)

        if isinstance(metric, type):
            print(metric_label, ' is a type, not instance. Forgot to initialize?')

        metric_func = copy.copy(metric)
        utils.fit_metric(metric_func, data, labels, sample_weight=weight)

        quality = OrderedDict()
        for estimator_name in self.prediction:
            if isinstance(steps, int):
                step = steps
            else:
                step = steps[estimator_name]
            try:
                quality[estimator_name] = self._learning_curve_additional(estimator_name, metric_func, step, mask,
                                                                          predict_only_masked=predict_only_masked)
            except (AttributeError, NotImplementedError):
                print("Estimator {} doesn't support stage predictions".format(estimator_name))
        plot_fig = plotting.FunctionsPlot(quality)
        plot_fig.xlabel = 'stage'
        plot_fig.ylabel = '{}'.format(metric_label)
        plot_fig.title = 'Learning curves'
        return plot_fig

    def _learning_curve_additional(self, name, metric_func, step, mask, predict_only_masked):
        """ returns tuple (x_values, quality_values), which describe the learning curve """
        raise NotImplementedError('Should be implemented in descendants')

    def feature_importance(self, grid_columns=2):
        """
        Get features importance

        :param int grid_columns: count of columns in grid
        :rtype: plotting.GridPlot
        """
        importance_plots = []
        for name, estimator in self.estimators.items():
            try:
                df = estimator.get_feature_importances()
                df = {column: dict(df[column]) for column in df.columns}
                plot = plotting.BarComparePlot(df, sortby='effect')
                plot.title = 'Feature importance for %s' % name
                plot.fontsize = 10
                importance_plots.append(plot)
            except AttributeError:
                print("Estimator {} doesn't support feature importances".format(name))
        return plotting.GridPlot(grid_columns, *importance_plots)

    def _feature_importance_shuffling(self, metric, mask=None, grid_columns=2):
        """
        Get features importance using shuffling method (apply random permutation to one particular column)

        :param metric: function to measure quality
            function(y_true, proba, sample_weight=None)
        :param mask: mask which points we should use
        :type mask: None or array-like or str or function(pandas.DataFrame)
        :param int grid_columns: number of columns in grid
        :rtype: plotting.GridPlot
        """
        importances_plots = []
        for name, estimator in self.estimators.items():
            result = dict()
            _, data, labels, weights = self._apply_mask(mask, self._get_features(estimator.features), self.target,
                                                        self.weight)
            metric_copy = copy.deepcopy(metric)
            utils.fit_metric(metric_copy, data, labels, sample_weight=weights)

            for feature in data.columns:
                data_modified = data.copy()
                column = numpy.array(data_modified[feature])
                numpy.random.shuffle(column)
                data_modified[feature] = column
                predictions = self._predict(estimator, data_modified)
                result[feature] = metric_copy(labels, predictions, sample_weight=weights)

            plot_fig = plotting.BarComparePlot({name: result}, sortby=name)
            plot_fig.title = 'Feature importance for %s' % name
            plot_fig.fontsize = 10
            importances_plots.append(plot_fig)
        return plotting.GridPlot(grid_columns, *importances_plots)

    def compute_metric(self, metric, mask=None):
        """
        Compute metric value

        :param metric: function like object with::

            __call__(self, y_true, prob, sample_weight=None)

        :param mask: mask, points we should use
        :type mask: None or array-like or str or function(pandas.DataFrame)

        :return: metric value for each estimator
        """
        mask, data, labels, weight = self._apply_mask(mask, self._get_features(), self.target, self.weight)

        if isinstance(metric, type):
            print('Metric is a type, not instance. Forgot to initialize?')
        metric_func = copy.copy(metric)
        utils.fit_metric(metric_func, data, labels, sample_weight=weight)

        quality = OrderedDict()
        for estimator_name, prediction in self.prediction.items():
            quality[estimator_name] = metric_func(labels, prediction[mask], sample_weight=weight)
        return quality