"""
This module contains report class for classification estimators. Report includes:

    * features scatter plots, distributions, correlations
    * learning curve
    * roc curve
    * efficiencies
    * metric vs cut
    * feature importance
    * feature importance by shuffling the feature column

All methods return objects, which have `plot` method (details see in :class:`rep.plotting`),
these objects contain raw information about things to be plotted.
"""

from __future__ import division, print_function, absolute_import
from itertools import islice
from collections import OrderedDict, defaultdict
import itertools

import numpy

from .. import utils
from .. import plotting
from ._base import AbstractReport
from .metrics import OptimalMetric, LogLoss
from ..estimators.interface import Classifier
from ..utils import get_columns_dict


__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'

BAR_TYPES = {'error_bar', 'bar'}


class ClassificationReport(AbstractReport):
    def __init__(self, classifiers, lds):
        """
        Test estimators on any data. Supports ROC curve, prediction distribution, features information
        (correlation matrix, distribution, scatter plots for pairs of features),
        efficiencies for thresholds (evaluate flatness of predictions for important feature),
        correlation with prediction for necessary feature, any metrics of quality.

        :param classifiers: estimators
        :type classifiers: dict[str, Classifier]
        :param LabeledDataStorage lds: data
        """

        for name, classifier in classifiers.items():
            assert isinstance(classifier, Classifier), "Object {} doesn't implement interface".format(name)

        AbstractReport.__init__(self, lds=lds, estimators=classifiers)

    def _predict(self, estimator, X):
        return estimator.predict_proba(X)

    @staticmethod
    def _check_labels(labels_dict, class_labels):
        """ Normalizes the names for labels.

        :param labels_dict: dict(label -> name) or None,
            if None, the classes will be named 0: bck and 1: signal
        :param class_labels: array of shape [n_samples] with labels of events,
            used here to define the set of used labels.
        """
        labels_dict_init = OrderedDict()
        all_classes = set(class_labels)
        if labels_dict is None:
            labels_dict_init[0] = 'bck'
            labels_dict_init[1] = 'signal'
        else:
            for key, value in labels_dict.items():
                if key in all_classes:
                    labels_dict_init[key] = value
        assert set(labels_dict_init.keys()).issubset(all_classes), \
            'Labels must be a subset of {}, but {}'.format(all_classes, list(labels_dict_init.keys()))
        return labels_dict_init

    def features_pdf(self, features=None, mask=None, bins=30, ignored_sideband=0.0, labels_dict=None, grid_columns=2):
        """
        Features distribution (with errors)

        :param features: using features (if None then use classifier's features)
        :type features: None or list[str]
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param bins: count of bins or array with boarders
        :type bins: int or array-like
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid
        :param float ignored_sideband: float from (0, 1), part of events ignored from the left and from the right
        :rtype: plotting.GridPlot
        """
        features = self.common_features if features is None else features
        pdf = defaultdict(OrderedDict)
        _, df, class_labels, weight = self._apply_mask(mask, self._get_features(features), self.target, self.weight)
        labels_dict = self._check_labels(labels_dict, class_labels)

        pdf_plots = []
        for feature in df.columns:
            for label, name in labels_dict.items():
                pdf[feature][name] = \
                    utils.calc_hist_with_errors(df[feature][class_labels == label].values,
                                                weight[class_labels == label], bins, ignored_sideband=ignored_sideband)
            plot_fig = plotting.ErrorPlot(pdf[feature])
            plot_fig.xlabel = feature
            plot_fig.ylabel = 'Normed event counts'
            plot_fig.figsize = (8, 6)
            pdf_plots.append(plot_fig)

        return plotting.GridPlot(grid_columns, *pdf_plots)

    def features_correlation_matrix_by_class(self, features=None, mask=None, tick_labels=None, vmin=-1, vmax=1,
                                             labels_dict=None, grid_columns=2):
        """
        Correlation between features (built separately for each class)

        :param features: using features (if None then use classifier's features)
        :type features: None or list[str]
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param tick_labels: names for features in matrix
        :type tick_labels: None or array-like
        :param int vmin: min of value for min color
        :param int vmax: max of value for max color
        :param int grid_columns: count of columns in grid

        :rtype: plotting.GridPlot
        """
        features = self.common_features if features is None else features
        _, df, class_labels = self._apply_mask(mask, self._get_features(features), self.target)
        features_names = list(df.columns)
        if tick_labels is None:
            tick_labels = features_names
        labels_dict = self._check_labels(labels_dict, class_labels)

        correlation_plots = []
        color_map = itertools.cycle(['Reds', 'Blues', 'Oranges'])
        for label, name in labels_dict.items():
            plot_corr = plotting.ColorMap(
                utils.calc_feature_correlation_matrix(df[features_names][class_labels == label]),
                labels=tick_labels, vmin=vmin, vmax=vmax, cmap=next(color_map))
            plot_corr.title = 'Correlation for %s events' % name
            plot_corr.fontsize = 10
            plot_corr.figsize = (len(features) // 5 + 2, len(features) // 5)
            correlation_plots.append(plot_corr)
        return plotting.GridPlot(grid_columns, *correlation_plots)

    def scatter(self, correlation_pairs, mask=None, marker_size=20, alpha=0.1, labels_dict=None, grid_columns=2):
        """
        Correlation between pairs of features

        :param list[tuple] correlation_pairs: pairs of features along which scatter plot will be build.
        :param mask: mask for data, which will be used
        :type mask: None or array-like or str or function(pandas.DataFrame)
        :param int marker_size: size of marker for each event on the plot
        :param float alpha: blending parameter for scatter
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid

        :rtype: plotting.GridPlot
        """
        features = list(set(itertools.chain.from_iterable(correlation_pairs)))

        _, df, class_labels = self._apply_mask(mask, self._get_features(features), self.target)
        labels_dict = self._check_labels(labels_dict, class_labels)

        correlation_plots = []
        corr_pairs = OrderedDict()
        for feature1_c, feature2_c in correlation_pairs:
            feature1, feature2 = list(get_columns_dict([feature1_c, feature2_c]).keys())
            corr_pairs[(feature1, feature2)] = OrderedDict()
            for label, name in labels_dict.items():
                corr_pairs[(feature1, feature2)][name] = (df[feature1][class_labels == label].values,
                                                          df[feature2][class_labels == label].values)
            plot_fig = plotting.ScatterPlot(corr_pairs[(feature1, feature2)], alpha=alpha, size=marker_size)
            plot_fig.xlabel = feature1
            plot_fig.ylabel = feature2
            plot_fig.figsize = (8, 6)
            correlation_plots.append(plot_fig)
        return plotting.GridPlot(grid_columns, *correlation_plots)

    def roc(self, mask=None, signal_label=1, physics_notion=False):
        """
        Calculate roc functions for data and return roc plot object

        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param bool physics_notion: if set to True, will show signal efficiency vs background rejection,
            otherwise TPR vs FPR.

        :rtype: plotting.FunctionsPlot
        """
        roc_curves = OrderedDict()
        mask, = self._apply_mask(mask)

        classes_labels = set(numpy.unique(self.target[mask]))
        assert len(classes_labels) == 2 and signal_label in classes_labels, \
            'Classes must be 2 instead of {}'.format(classes_labels)

        for name, prediction in self.prediction.items():
            labels_active = numpy.array(self.target[mask] == signal_label, dtype=int)
            (tpr, tnr), _, _ = utils.calc_ROC(prediction[mask, signal_label], labels_active,
                                              sample_weight=self.weight[mask])
            if physics_notion:
                roc_curves[name] = (tpr, tnr)
                xlabel = 'Signal sensitivity'
                ylabel = 'Bg rejection eff (specificity)'
            else:
                roc_curves[name] = (1 - tnr, tpr)
                xlabel = 'false positive rate'
                ylabel = 'true positive rate'

        plot_fig = plotting.FunctionsPlot(roc_curves)
        plot_fig.xlabel = xlabel
        plot_fig.ylabel = ylabel
        plot_fig.title = 'ROC curves'
        return plot_fig

    def prediction_pdf(self, mask=None, target_class=1, bins=30, size=2, log=False, plot_type='error_bar',
                       normed=True, labels_dict=None):
        """
        Distribution of prediction for signal and bck separately with errors

        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param target_class: draw probabilities of being classified as target_class
            (default 1, will draw signal probabilities).
            If None, will draw probability corresponding to right class of each event.
        :type target_class: int or None
        :param bins: number of bins in histogram
        :type bins: int or array-like
        :param int size: points size on plots
        :param bool log: use logarithmic scale
        :param bool normed: draw normed pdf or not (normed by default)
        :param str plot_type: 'error_bar' for error type and 'bar' for hist type
        :param labels_dict: names for class labels as dictionary
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :rtype: plotting.ErrorPlot or plotting.BarPlot
        """
        assert plot_type in BAR_TYPES, 'Value for plot_type must be in ' + str(BAR_TYPES)
        data = OrderedDict()
        mask, = self._apply_mask(mask)
        class_labels, weight = self.target[mask], self.weight[mask]
        labels_dict = self._check_labels(labels_dict, class_labels)

        filled_type = itertools.cycle(['not_filled', 'filled'])
        for name, prediction in self.prediction.items():
            prediction = prediction[mask]
            for label, name_label in labels_dict.items():
                label_mask = class_labels == label
                target_label = label if target_class is None else target_class
                plot_name = '{name} for {cl}'.format(name=name_label, cl=name)
                if plot_type == 'error_bar':
                    data[plot_name] = utils.calc_hist_with_errors(
                        prediction[label_mask, target_label],
                        weight[label_mask], bins, normed=normed, x_range=(0, 1))
                else:
                    data[plot_name] = (prediction[label_mask, target_label], weight[label_mask], next(filled_type))

        if plot_type == 'error_bar':
            plot_fig = plotting.ErrorPlot(data, size=size, log=log)
        else:
            plot_fig = plotting.BarPlot(data, bins=bins, normalization=normed, value_range=(0, 1))
        plot_fig.xlabel = 'prediction'
        plot_fig.ylabel = 'density' if normed else 'Event count'
        return plot_fig

    def efficiencies(self, features, thresholds=None, mask=None, bins=30, labels_dict=None, ignored_sideband=0.0,
                     errors=False, grid_columns=2):
        """
        Efficiencies for spectators

        :param features: using features (if None then use classifier's spectators)
        :type features: None or list[str]
        :param bins: bins for histogram
        :type bins: int or array-like
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param list[float] thresholds: thresholds on prediction
        :param bool errors: if True then use errorbar, else interpolate function
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid
        :param float ignored_sideband: (0, 1) percent of plotting data

        :rtype: plotting.GridPlot
        """
        mask, data, class_labels, weight = self._apply_mask(
            mask, self._get_features(features), self.target, self.weight)
        labels_dict = self._check_labels(labels_dict, class_labels)

        plots = []
        for feature in data.columns:
            for name, prediction in self.prediction.items():
                prediction = prediction[mask]
                eff = OrderedDict()
                for label, label_name in labels_dict.items():
                    label_mask = class_labels == label
                    eff[label_name] = utils.get_efficiencies(prediction[label_mask, label],
                                                             data[feature][label_mask].values,
                                                             bins_number=bins,
                                                             sample_weight=weight[label_mask],
                                                             thresholds=thresholds, errors=errors,
                                                             ignored_sideband=ignored_sideband)

                for label_name, eff_data in eff.items():
                    if errors:
                        plot_fig = plotting.ErrorPlot(eff_data)
                    else:
                        plot_fig = plotting.FunctionsPlot(eff_data)
                    plot_fig.xlabel = feature
                    plot_fig.ylabel = 'Efficiency for {}'.format(name)
                    plot_fig.title = '{} flatness'.format(label_name)
                    plot_fig.ylim = (0, 1)
                    plots.append(plot_fig)

        return plotting.GridPlot(grid_columns, *plots)

    def metrics_vs_cut(self, metric, mask=None, metric_label='metric'):
        """
        Draw values of binary metric depending on the threshold on predictions.

        :param metric: binary metric (AMS, f1 or so - shall use only tpr and fpr)
        :param mask: mask for data used in comparison
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param str metric_label: name for metric on plot

        :rtype: plotting.FunctionsPlot
        """

        mask, = self._apply_mask(mask)
        class_labels, weight = self.target[mask], self.weight[mask]

        # assert len(numpy.unique(class_labels)) == 2, 'This function supports only for 2-classification'

        quality = OrderedDict()
        opt_metrics = OptimalMetric(metric)
        for classifier_name, prediction in self.prediction.items():
            prediction = prediction[mask]
            quality[classifier_name] = opt_metrics.compute(class_labels, prediction, weight)
        plot_fig = plotting.FunctionsPlot(quality)
        plot_fig.xlabel = 'predictions thresholds'
        plot_fig.ylabel = metric_label
        return plot_fig

    def _learning_curve_additional(self, name, metric_func, step, mask, predict_only_masked):
        """
        Compute values of RocAuc (or some other metric) for particular classifier, mask and metric function.
        :return: tuple(stages, values) with numbers of stages and corresponding
        computed values of metric after each stage.
        """

        evaled_mask, labels, weight = self._apply_mask(mask, self.target, self.weight)
        data = self._get_features()
        if predict_only_masked:
            _, data = self._apply_mask(mask, data)

        curve = OrderedDict()
        stage_proba = self.estimators[name].staged_predict_proba(data)
        for stage, prediction in islice(enumerate(stage_proba), step - 1, None, step):
            if not predict_only_masked:
                prediction = prediction[evaled_mask]
            curve[stage] = metric_func(labels, prediction, sample_weight=weight)
        return list(curve.keys()), list(curve.values())

    def feature_importance_shuffling(self, metric=LogLoss(), mask=None, grid_columns=2):
        """
        Get features importance using shuffling method (apply random permutation to one particular column)

        :param metric: function to measure quality
            function(y_true, proba, sample_weight=None)
        :param mask: mask which points the data we should train on
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param int grid_columns: number of columns in grid
        :rtype: plotting.GridPlot
        """
        return self._feature_importance_shuffling(metric=metric, mask=mask, grid_columns=grid_columns)

    @staticmethod
    def _compute_bin_indices(columns, bin_limits):
        """ Compute bin indices. For each axis, first and last value are ignored """
        assert len(columns) == len(bin_limits), 'Indices are the same'
        bin_indices = numpy.zeros(len(columns[0]), dtype=int)
        for column, axis_limits in zip(columns, bin_limits):
            bin_indices *= len(axis_limits) - 1
            axis_indices = numpy.searchsorted(axis_limits[1:-1], column)
            bin_indices += axis_indices
        return bin_indices

    def efficiencies_2d(self, features, efficiency, mask=None, n_bins=20, ignored_sideband=0.0, labels_dict=None,
                        grid_columns=2, signal_label=1, cmap='RdBu'):
        """
        For binary classification plots the dependence of efficiency on two columns

        :param features: tuple of list with names of two features
        :param float efficiency: efficiency, float
        :param n_bins: bins for histogram
        :type n_bins: int or array-like
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid
        :param float ignored_sideband: (0, 1) percent of plotting data
        :param int signal_label: label to calculate efficiency threshold
        :param str cmap: name of colormap used

        :rtype: plotting.GridPlot
        """

        assert len(features) == 2, 'you should provide two columns'
        assert 0. <= efficiency <= 1., 'efficiency should be in range (0, 1)'

        mask, data, class_labels, weight = self._apply_mask(
            mask, self._get_features(features), self.target, self.weight)
        labels_dict = self._check_labels(labels_dict, class_labels)

        plots = []
        columns = []
        axis_limits = []
        bin_limits = []
        bin_centers = []

        for feature in data.columns:
            column = numpy.array(data[feature])
            columns.append(column)
            axis_min, axis_max = numpy.percentile(column, [100 * ignored_sideband, 100 * (1. - ignored_sideband)])
            axis_limits.append([axis_min, axis_max])
            bin_limits.append(numpy.linspace(axis_min, axis_max, n_bins + 1))
            bin_centers.append(numpy.linspace(axis_min, axis_max, 2 * n_bins + 1)[1::2])
            assert len(bin_limits[-1]) == n_bins + 1
            assert len(bin_centers[-1]) == n_bins
        columns_labels = tuple(data.columns)
        bin_indices = self._compute_bin_indices(columns, bin_limits=bin_limits)

        sig_mask = class_labels == signal_label
        for classifier_name, prediction in self.prediction.items():
            prediction = prediction[mask]
            threshold_ = utils.weighted_quantile(prediction[sig_mask, signal_label], (1. - efficiency))
            passed = prediction[:, signal_label] > threshold_
            minlength = n_bins ** 2
            for label, label_name in labels_dict.items():
                # recompute threshold
                label_mask = class_labels == label
                assert numpy.all(bin_indices < minlength)

                # mean efficiency
                mean_eff = numpy.sum(label_mask * weight * passed) / numpy.sum(label_mask * weight)

                bin_efficiencies = numpy.bincount(bin_indices, weights=label_mask * weight * passed, minlength=minlength)
                denominators = numpy.bincount(bin_indices, weights=label_mask * weight, minlength=minlength)
                bin_efficiencies /= denominators + 1e-6
                # For empty bins we will return mean (plots otherwise become ugly)
                bin_efficiencies[denominators == 0] = mean_eff

                plot_fig = plotting.Function2D_Plot(lambda x, y: 0, xlim=axis_limits[0], ylim=axis_limits[1])
                plot_fig.x, plot_fig.y = numpy.meshgrid(*bin_centers)
                plot_fig.z = bin_efficiencies.reshape([n_bins, n_bins])
                plot_fig.xlabel, plot_fig.ylabel = columns_labels
                plot_fig.title = 'Estimator {} efficiencies for class {}'.format(classifier_name, label_name)
                plot_fig.vmin = mean_eff - 0.2
                plot_fig.vmax = mean_eff + 0.2
                plot_fig.cmap = cmap
                plots.append(plot_fig)

        return plotting.GridPlot(grid_columns, *plots)
