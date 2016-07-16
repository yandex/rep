"""
There are different plotting backends supported:

    * *matplotlib* (default, de-facto standard plotting library in python),
    * *ROOT* (the library used by CERN people),
    * *bokeh* (open-source package with interactive plots)

"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import itertools
from IPython import get_ipython

import matplotlib.pyplot as plt
import numpy
import tempfile
from IPython.core import display

COLOR_ARRAY = ['red', 'blue', 'green', 'cyan', 'MediumVioletRed', 'k', 'navy', 'lime', 'CornflowerBlue',
               "coral", 'DeepPink', 'LightBlue', 'yellow', 'Purple', 'YellowGreen', 'magenta']
COLOR_ARRAY_BOKEH = ['red', 'blue', 'green', 'cyan']
COLOR_ARRAY_TMVA = range(2, 10)
BOKEH_CMAP = [
    '#75968f', '#a5bab7', '#c9d9d3', '#e2e2e2', '#dfccce',
    '#ddb7b1', '#cc7878', '#933b41', '#550b1d'
]

_COLOR_CYCLE = itertools.cycle(COLOR_ARRAY)
_COLOR_CYCLE_BOKEH = itertools.cycle(COLOR_ARRAY_BOKEH)
_COLOR_CYCLE_TMVA = itertools.cycle(COLOR_ARRAY_TMVA)
_BOKEH_OUTPUT_NOTEBOOK_ACTIVATED = False

__author__ = 'Tatiana Likhomanenko'


class AbstractPlot(object):
    """
    Abstract class for possible plot objects, which implements plot function.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.BOKEH_RESIZE = 50
        self.TMVA_RESIZE = 80
        self.xlim = None
        self.ylim = None
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.figsize = (13, 7)
        self.fontsize = 14
        self.new_plot = False
        self.canvas = None
        self._tmva_keeper = []

    @abstractmethod
    def _plot(self):
        pass

    @abstractmethod
    def _plot_tmva(self):
        pass

    @abstractmethod
    def _plot_bokeh(self, current_plot, show_legend=True):
        pass

    def _repr_html_(self):
        """Representation for IPython"""
        self.plot()
        return ""

    def plot(self, new_plot=False, xlim=None, ylim=None, title=None, figsize=None,
             xlabel=None, ylabel=None, fontsize=None, show_legend=True, grid=True):
        """
        Plot data using matplotlib library. Use show() method for matplotlib to see result or ::

            %matplotlib inline

        in IPython to see plot as cell output.

        :param bool new_plot: create or not new figure
        :param xlim: x-axis range
        :param ylim: y-axis range
        :type xlim: None or tuple(x_min, x_max)
        :type ylim: None or tuple(y_min, y_max)
        :param title: title
        :type title: None or str
        :param figsize: figure size
        :type figsize: None or tuple(weight, height)
        :param xlabel: x-axis name
        :type xlabel: None or str
        :param ylabel: y-axis name
        :type ylabel: None or str
        :param fontsize: font size
        :type fontsize: None or int
        :param bool show_legend: show or not labels for plots
        :param bool grid: show grid or not

        """
        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        figsize = self.figsize if figsize is None else figsize
        fontsize = self.fontsize if fontsize is None else fontsize
        self.fontsize_ = fontsize
        self.show_legend_ = show_legend
        title = self.title if title is None else title
        xlim = self.xlim if xlim is None else xlim
        ylim = self.ylim if ylim is None else ylim
        new_plot = self.new_plot or new_plot

        if new_plot:
            plt.figure(figsize=figsize)

        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        plt.grid(grid)

        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        self._plot()

        if show_legend:
            plt.legend(loc='best', scatterpoints=1)

    def plot_bokeh(self, xlim=None, ylim=None, title=None, figsize=None,
                   xlabel=None, ylabel=None, fontsize=None, show_legend=True):

        """
        Plot data using bokeh library. Use show() method for bokeh to see result.

        :param xlim: x-axis range
        :param ylim: y-axis range
        :type xlim: None or tuple(x_min, x_max)
        :type ylim: None or tuple(y_min, y_max)
        :param title: title
        :type title: None or str
        :param figsize: figure size
        :type figsize: None or tuple(weight, height)
        :param xlabel: x-axis name
        :type xlabel: None or str
        :param ylabel: y-axis name
        :type ylabel: None or str
        :param fontsize: font size
        :type fontsize: None or int
        :param bool show_legend: show or not labels for plots
        """
        global _COLOR_CYCLE_BOKEH
        global _BOKEH_OUTPUT_NOTEBOOK_ACTIVATED
        import bokeh.plotting as bkh
        from bokeh.models import Range1d
        from bokeh.core.properties import value

        figsize = self.figsize if figsize is None else figsize
        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        title = self.title if title is None else title
        xlim = self.xlim if xlim is None else xlim
        ylim = self.ylim if ylim is None else ylim
        fontsize = self.fontsize if fontsize is None else fontsize
        self.fontsize_ = fontsize
        self.show_legend_ = show_legend

        figsize = (figsize[0] * self.BOKEH_RESIZE, figsize[1] * self.BOKEH_RESIZE)

        if not _BOKEH_OUTPUT_NOTEBOOK_ACTIVATED:
            bkh.output_notebook()
            _BOKEH_OUTPUT_NOTEBOOK_ACTIVATED = True

        current_plot = bkh.figure(title=title, plot_width=figsize[0], plot_height=figsize[1])
        _COLOR_CYCLE_BOKEH = itertools.cycle(COLOR_ARRAY_BOKEH)

        if xlim is not None:
            current_plot.x_range = Range1d(start=xlim[0], end=xlim[1])
        if ylim is not None:
            current_plot.y_range = Range1d(start=ylim[0], end=ylim[1])
        current_plot.title_text_font_size = value("{}pt".format(fontsize))
        current_plot.xaxis.axis_label = xlabel
        current_plot.yaxis.axis_label = ylabel
        current_plot.legend.orientation = 'top_right'

        current_plot = self._plot_bokeh(current_plot, show_legend)
        bkh.show(current_plot)

    def plot_tmva(self, new_plot=False, style_file=None, figsize=None,
                  xlim=None, ylim=None, title=None, xlabel=None, ylabel=None, show_legend=True):
        """
        Plot data using tmva library.

        :param bool new_plot: create or not new figure
        :param style_file: tmva styles configuring file
        :type style_file: None or str
        :param xlim: x-axis range
        :param ylim: y-axis range
        :type xlim: None or tuple(x_min, x_max)
        :type ylim: None or tuple(y_min, y_max)
        :param title: title
        :type title: None or str
        :param figsize: figure size
        :type figsize: None or tuple(weight, height)
        :param xlabel: x-axis name
        :type xlabel: None or str
        :param ylabel: y-axis name
        :type ylabel: None or str
        :param bool show_legend: show or not labels for plots
        """
        import ROOT

        global _COLOR_CYCLE_TMVA

        self._tmva_keeper = []
        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        figsize = self.figsize if figsize is None else figsize
        title = self.title if title is None else title
        xlim = self.xlim if xlim is None else xlim
        ylim = self.ylim if ylim is None else ylim

        if new_plot or self.canvas is None:
            _COLOR_CYCLE_TMVA = itertools.cycle(COLOR_ARRAY_TMVA)
            t = numpy.random.randint(low=100, high=100000)
            figsize = (figsize[0] * self.TMVA_RESIZE, figsize[1] * self.TMVA_RESIZE)
            self.canvas = canvas("canvas{}".format(t), figsize)

        if style_file is not None:
            ROOT.gROOT.LoadMacro(style_file)
        else:
            self.canvas.SetFillColor(0)
            self.canvas.SetGrid()
            self.canvas.GetFrame().SetFillColor(21)
            self.canvas.GetFrame().SetBorderSize(12)

        graph, leg = self._plot_tmva()

        graph.SetTitle(title)
        graph.GetXaxis().SetTitle(xlabel)
        if xlim is not None:
            graph.GetXaxis().SetLimits(xlim[0], xlim[1])
        graph.GetYaxis().SetTitle(ylabel)
        if ylim is not None:
            graph.SetMinimum(ylim[0])
            graph.SetMaximum(ylim[1])
        if show_legend:
            leg.Draw()
        self._tmva_keeper.append((graph, leg))
        return self.canvas


class GridPlot(AbstractPlot):
    def __init__(self, columns=3, *plots):
        """
        Implements grid of plots (set of plots organized in a grid).

        :param int columns: count of columns in grid
        :param list[AbstractPlot] plots: plot objects
        """
        super(GridPlot, self).__init__()
        self.plots = plots
        self.columns = columns
        self.rows = (len(plots) + self.columns - 1) // self.columns
        width = max([elem.figsize[0] for elem in self.plots])
        height = max([elem.figsize[1] for elem in self.plots])
        self.figsize = (self.columns * width, self.rows * height)
        self.one_figsize = (width, height)
        self.new_plot = True

    def _plot(self):
        for i, plotter in enumerate(self.plots):
            plt.subplot(self.rows, self.columns, i + 1)
            plotter.plot(fontsize=self.fontsize_, show_legend=self.show_legend_)

    def _plot_bokeh(self, current_plot, show_legend=True):
        from bokeh import models
        import bokeh.plotting as bkh
        from bokeh.core.properties import value

        lst = []
        row_lst = []
        for plotter in self.plots:
            cur_plot = bkh.figure(title=plotter.title, plot_width=self.one_figsize[0] * self.BOKEH_RESIZE,
                                  plot_height=self.one_figsize[1] * self.BOKEH_RESIZE)
            if plotter.xlim is not None:
                cur_plot.x_range = models.Range1d(start=plotter.xlim[0], end=plotter.xlim[1])
            if plotter.ylim is not None:
                cur_plot.y_range = models.Range1d(start=plotter.ylim[0], end=plotter.ylim[1])
            cur_plot.title_text_font_size = value("{}pt".format(plotter.fontsize))
            cur_plot.xaxis.axis_label = plotter.xlabel
            cur_plot.yaxis.axis_label = plotter.ylabel
            cur_plot.legend.orientation = 'top_right'
            cur_plot = plotter._plot_bokeh(cur_plot, show_legend=show_legend)
            if len(row_lst) >= self.columns:
                lst.append(row_lst)
                row_lst = []
            row_lst.append(cur_plot)
        if len(row_lst) > 0:
            lst.append(row_lst)
        grid = models.GridPlot(children=lst)
        return grid

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")


class HStackPlot(AbstractPlot):
    def __init__(self, *plots):
        """
        Horizontal stack of plots.

        :param list[AbstractPlot] plots: plot objects
        """
        super(HStackPlot, self).__init__()
        self.plots = plots
        width = sum([elem.figsize[0] for elem in self.plots])
        height = max([elem.figsize[1] for elem in self.plots])
        self.figsize = (width, height)
        self.new_plot = True

    def _plot(self):
        for i, plotter in enumerate(self.plots):
            plt.subplot(1, len(self.plots), i + 1)
            plotter.plot(fontsize=self.fontsize_, show_legend=self.show_legend_)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        obj = GridPlot(len(self.plots), *self.plots)
        return obj._plot_bokeh(current_plot, show_legend=show_legend)


class VStackPlot(AbstractPlot):
    def __init__(self, *plots):
        """
        Implements vertical stack plots

        :param list[AbstractPlot] plots: plot objects
        """
        super(VStackPlot, self).__init__()
        self.plots = plots

    def _plot(self):
        for i, plotter in enumerate(self.plots):
            plt.subplot(len(self.plots), 1, i + 1)
            plotter.plot(fontsize=self.fontsize_, show_legend=self.show_legend_)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        obj = GridPlot(1, *self.plots)
        return obj._plot_bokeh(current_plot, show_legend=show_legend)


class ErrorPlot(AbstractPlot):
    def __init__(self, errors, size=2, log=False):
        """
        Implements error bars plots

        :param errors: name - x points, y points, y errors, x errors
        :type errors: dict[str, tuple(array, array, array, array)]
        :param int size: size of scatters
        :param bool log: logarithm scaling
        """
        super(ErrorPlot, self).__init__()
        self.errors = errors
        self.size = size
        self.log = log

    def _plot(self):
        for name, val in self.errors.items():
            x, y, y_err, x_err = val
            y_err_mod = y_err
            if self.log:
                y_mod = numpy.log(y)
                if y_err is not None:
                    y_err_mod = numpy.log(y + y_err) - y_mod
            else:
                y_mod = y
            err_bar = plt.errorbar(x, y_mod, yerr=y_err_mod, xerr=x_err, label=name, fmt='o', ms=self.size)
            err_bar[0].set_label('_nolegend_')

    def _plot_tmva(self):
        import ROOT

        multigraph = ROOT.TMultiGraph()
        legend = ROOT.TLegend(0.2, 0.2, 0.5, 0.4)
        for name, val in self.errors.items():
            color = next(_COLOR_CYCLE_TMVA)
            x, y, yerr, xerr = val
            gr = ROOT.TGraphErrors(len(x), numpy.array(x), numpy.array(y), numpy.array(xerr), numpy.array(yerr))
            gr.SetDrawOption("AP")
            gr.SetMarkerColor(color)
            gr.SetMarkerStyle(1)
            gr.SetFillStyle(0)
            gr.SetLineColor(color)
            gr.SetLineWidth(0)
            multigraph.Add(gr)
            legend.AddEntry(gr, name)
        multigraph.Draw("AP")
        return multigraph, legend

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class FunctionsPlot(AbstractPlot):
    def __init__(self, functions):
        """
        Implements 1d-function plots

        :param functions: dict which maps label of curve to x, y coordinates of points
        :type functions: dict[str, tuple(array, array)]
        """
        super(FunctionsPlot, self).__init__()
        self.functions = functions

    def _plot(self):
        for name, data_xy in self.functions.items():
            x_val, y_val = data_xy
            plt.plot(x_val, y_val, linewidth=2, label=name)

    def _plot_bokeh(self, current_plot, show_legend=True):
        for name, data_xy in self.functions.items():
            color = next(_COLOR_CYCLE_BOKEH)
            x_val, y_val = data_xy
            legend_name = None
            if show_legend:
                legend_name = name
            current_plot.line(x_val, y_val, line_width=2, legend=legend_name, color=color)
        return current_plot

    def _plot_tmva(self):
        import ROOT

        multigraph = ROOT.TMultiGraph()
        legend = ROOT.TLegend(0.2, 0.2, 0.5, 0.4)
        for name, data_xy in self.functions.items():
            color = next(_COLOR_CYCLE_TMVA)
            x_val, y_val = data_xy
            gr = ROOT.TGraph(len(x_val), numpy.array(x_val), numpy.array(y_val))
            gr.SetTitle(name)
            gr.SetMarkerColor(color)
            gr.SetMarkerStyle(2)
            gr.SetDrawOption("APL")
            gr.SetLineColor(color)
            gr.SetLineWidth(1)
            gr.SetFillStyle(0)
            multigraph.Add(gr)
            legend.AddEntry(gr, name, "L")
        multigraph.Draw("AC")
        return multigraph, legend


class ColorMap(AbstractPlot):
    def __init__(self, matrix, labels=None, cmap='jet', vmin=-1, vmax=1):
        """
        Implements color map plots

        :param numpy.ndarray matrix: matrix
        :param labels: names for each matrix-row
        :type labels: None or list[str]
        :param str cmap: color map name
        :param float vmin: min value for color map
        :param float vmax: max value for color map
        """
        super(ColorMap, self).__init__()
        self.matrix = matrix
        self.labels = labels if labels is not None else range(matrix.shape[0])
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def _plot(self):
        p = plt.pcolor(self.matrix, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        plt.colorbar(p)
        plt.xlim((0, self.matrix.shape[0]))
        plt.ylim((0, self.matrix.shape[1]))
        if self.labels is not None:
            plt.xticks(numpy.arange(0.5, len(self.labels) + 0.5), self.labels, fontsize=self.fontsize, rotation=90)
            plt.yticks(numpy.arange(0.5, len(self.labels) + 0.5), self.labels, fontsize=self.fontsize)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        from bokeh.models.tools import HoverTool
        from collections import OrderedDict
        from bokeh.models.ranges import FactorRange
        import bokeh.plotting as bkh

        value_lst = self.matrix.flatten()
        min_el = min(value_lst)
        vmax = float(max(value_lst) - min_el)
        color_lst = [BOKEH_CMAP[int((v - min_el) / vmax * (len(BOKEH_CMAP) - 1))] \
                     for v in value_lst]

        source = bkh.ColumnDataSource(
            data=dict(
                x=self.labels * self.matrix.shape[0],
                y=numpy.array([[i] * self.matrix.shape[1] for i in self.labels]).flatten(),
                color=color_lst,
                value=value_lst,
            )
        )
        # current_plot._below = []
        current_plot.x_range = FactorRange(factors=self.labels)
        current_plot.y_range = FactorRange(factors=self.labels)
        # current_plot._left = []

        # current_plot.extra_y_ranges = {"foo": bkh.FactorRange(factors=self.labels)}
        # current_plot.add_layout(CategoricalAxis(y_range_name="foo"), place='left')
        # current_plot.extra_x_ranges = {"foo": bkh.FactorRange(factors=self.labels)}
        # current_plot.add_layout(CategoricalAxis(x_range_name="foo"), place='top')

        current_plot.rect('x', 'y', 0.98, 0.98, source=source, color='color', line_color=None)
        current_plot.grid.grid_line_color = None
        current_plot.axis.axis_line_color = None
        current_plot.axis.major_tick_line_color = None
        hover = current_plot.select(dict(type=HoverTool))
        if not hover:
            hover = HoverTool(plot=current_plot)
        hover.tooltips = OrderedDict([
            ('labels', '@x @y'),
            ('value', '@value')
        ])
        current_plot.tools.append(hover)
        return current_plot


class ScatterPlot(AbstractPlot):
    def __init__(self, scatters, alpha=0.1, size=20):
        """
        Implements scatters plots

        :param scatters: name - x points, y points
        :type scatters: dict[str, tuple(array, array)]
        :param int size: scatters size
        :param float alpha: transparency
        """

        super(ScatterPlot, self).__init__()
        self.scatters = scatters
        self.alpha = alpha
        self.size = size

    def _plot(self):
        sum_elements = sum([len(scatter[0]) for scatter in self.scatters.values()])
        for name, scatter in self.scatters.items():
            alpha_normed = min(float(self.alpha) / len(scatter[0]) * sum_elements, 1.)
            plt.scatter(scatter[0], scatter[1], s=self.size, c=next(_COLOR_CYCLE),
                        alpha=alpha_normed, label=name)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        for i, (name, scatter) in enumerate(self.scatters.items()):
            color = next(_COLOR_CYCLE_BOKEH)
            x_val, y_val = scatter
            legend_name = None
            if show_legend:
                legend_name = name
            current_plot.scatter(x_val, y_val, size=self.size // 5, alpha=self.alpha, legend=legend_name, color=color)
        return current_plot


class BarPlot(AbstractPlot):
    def __init__(self, data, bins=30, normalization=True, value_range=None):
        """
        Implements bar plots

        :param data: name - value, weight, style ('filled', another)
        :type data: dict[str, tuple(array, array, str)]
        :param bins: bins for histogram
        :type bins: int or list[float]
        :param bool normalization: normalize to pdf histogram or not
        :param value_range: min and max values
        :type value_range: None or tuple
        """
        super(BarPlot, self).__init__()
        self.data = data
        self.bins = bins
        self.normalization = normalization
        self.value_range = value_range

    def _plot(self):
        for label, sample in self.data.items():
            color = next(_COLOR_CYCLE)
            prediction, weight, style = sample
            if self.value_range is None:
                c_min, c_max = numpy.min(prediction), numpy.max(prediction)
            else:
                c_min, c_max = self.value_range
            histo = numpy.histogram(prediction, bins=self.bins, range=(c_min, c_max), weights=weight)
            norm = 1.0
            if self.normalization:
                norm = float(self.bins) / (c_max - c_min) / sum(weight)
            bin_edges = histo[1]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
            bin_widths = (bin_edges[1:] - bin_edges[:-1])

            yerr = []
            for i in range(len(bin_edges) - 1):
                weight_bin = weight[(prediction > bin_edges[i]) * (prediction <= bin_edges[i + 1])]
                yerr.append(numpy.sqrt(sum(weight_bin * weight_bin)) * norm)

            if style == 'filled':
                plt.bar(bin_centers - bin_widths / 2., numpy.array(histo[0]) * norm, facecolor=color,
                        linewidth=0, width=bin_widths, label=label, alpha=0.5)
            else:
                plt.bar(bin_centers - bin_widths / 2., norm * numpy.array(histo[0]),
                        edgecolor=color, color=color, ecolor=color, linewidth=1,
                        width=bin_widths, label=label, alpha=0.5, hatch="/", fill=False)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class BarComparePlot(AbstractPlot):
    def __init__(self, data, alpha=0.5, sortby=None, step=5):
        """
        Implements bar plots

        :param data:
        :type data: dict[str, dict(str, float)]
        :param float alpha: opacity
        :param sortby: sort bars by this data key
        :type sortby: None or str
        :param int step: length
        """
        super(BarComparePlot, self).__init__()
        self.data = data
        self.alpha = alpha
        self.sortby = sortby
        self.step = step

    def _plot(self):
        length = len(self.data) + self.step
        if self.sortby is not None:
            inds = numpy.argsort(list(self.data[self.sortby].values()))[::-1]
        else:
            inds = numpy.array(range(len(self.data[list(self.data.keys())[0]])))
        xticks_labels = numpy.array(list(self.data[list(self.data.keys())[0]].keys()))[inds]
        for move, (label, sample) in enumerate(self.data.items()):
            color = next(_COLOR_CYCLE)
            index = numpy.arange(len(sample))
            plt.bar(length * index + move, numpy.array(list(sample.values()))[inds], 1., alpha=self.alpha, color=color,
                    label=label)

        plt.xticks(length * numpy.arange(len(inds)), xticks_labels, rotation=90)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        length = len(self.data) + self.step
        if self.sortby is not None:
            inds = numpy.argsort(list(self.data[self.sortby].values()))[::-1]
        else:
            inds = numpy.array(range(len(self.data[list(self.data.keys())[0]])))

        # xticks_labels = numpy.array(list(self.data[list(self.data.keys())[0]].keys()))[inds]

        for move, (label, sample) in enumerate(self.data.items()):
            color = next(_COLOR_CYCLE_BOKEH)
            index = numpy.arange(len(sample))
            legend_name = None
            if show_legend:
                legend_name = label
            current_plot.rect(x=length * index + move, y=numpy.array(list(sample.values()))[inds] / 2,
                              height=numpy.array(list(sample.values()))[inds], width=1.,
                              alpha=self.alpha, color=color,
                              legend=legend_name)
        return current_plot


class Function2D_Plot(AbstractPlot):
    def __init__(self, function, xlim, ylim, xsteps=100, ysteps=100, cmap='Blues',
                 vmin=None, vmax=None):
        """
        Implements 2d-functions plots

        :param function function: vector function (X, Y)
        :param tuple(float, float) xlim: x ranges
        :param tuple(float, float) ylim: y ranges
        :param int xsteps: count of points for approximation on x-axis
        :param int ysteps: count of points for approximation on y-axis
        :param str cmap: color map
        :param float vmin: value, corresponding to minimum on cmap
        :param float vmax: value, corresponding to maximum on cmap
        """
        super(Function2D_Plot, self).__init__()

        x = numpy.linspace(xlim[0], xlim[1], xsteps)
        y = numpy.linspace(ylim[0], ylim[1], ysteps)
        self.xlim = xlim
        self.ylim = ylim
        self.x, self.y = numpy.meshgrid(x, y)
        self.z = function(self.x, self.y)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def _plot(self):
        colormap = plt.pcolor(self.x, self.y, self.z, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        cb = plt.colorbar(colormap)
        cb.set_label('value')

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported for bokeh")


class Histogram2D_Plot(AbstractPlot):
    def __init__(self, data, bins=30, cmap='Blues', cmin=None, cmax=None, range=None, normed=False):
        """
        Implements correlations plots

        :param (array, array) data: name var, name var - values for first, values for second
        :param bins: count of bins
        :type bins: int or list[float]
        :param str cmap: color map
        :param float cmin: value, corresponding to minimum on cmap
        :param float cmax: value, corresponding to maximum on cmap
        :param bool normed: normalize histogram
        :param range: array_like shape(2, 2), optional, default: None
            [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
            considered outliers and not tallied in the histogram.
        """
        super(Histogram2D_Plot, self).__init__()
        self.data = data
        self.binsX, self.binsY = (bins, bins) if isinstance(bins, int) else bins
        self.cmap = cmap
        self.vmin = cmin
        self.vmax = cmax
        self.range = range
        self.normed = normed

    def _plot(self):
        X, Y = self.data
        _, _, _, colormap = plt.hist2d(X, Y, bins=(self.binsX, self.binsY), range=self.range, normed=self.normed,
                                       cmin=self.vmin, cmax=self.vmax, cmap=self.cmap)
        cb = plt.colorbar(colormap)
        cb.set_label('value')

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class CorrelationPlot(AbstractPlot):
    def __init__(self, data, bins=30):
        """
        Implements correlations plots

        :param (array, array) data: values for first, values for second
        :param bins: count of bins
        :type bins: int or list[float]

        """
        super(CorrelationPlot, self).__init__()
        self.data = data
        self.bins = bins

    def _plot(self):
        (binsX, binsY) = (self.bins, self.bins) if isinstance(self.bins, int) else self.bins
        X, Y = self.data
        H, ex, ey = numpy.histogram2d(X, Y, bins=(binsX, binsY))
        x_center = numpy.diff(ex) / 2 + ex[0:-1]
        x_digit = numpy.digitize(X, ex)
        y_center = numpy.empty(binsY)
        y_std = numpy.empty(binsY)
        for i in range(binsX):
            y_pop = Y[numpy.where(x_digit == i + 1)[0]]
            y_center[i] = numpy.mean(y_pop)
            y_std[i] = numpy.std(y_pop)
        plt.errorbar(x_center, y_center, y_std)
        plt.xlim(ex[0], ex[-1])

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class CorrelationMapPlot(AbstractPlot):
    def __init__(self, data, bins=30):
        """
        Implements correlations map plots

        :param (array, array) data: name var, name var - values for first, values for second
        :param bins: count of bins
        :type bins: int or list[float]
        """
        super(CorrelationMapPlot, self).__init__()
        self.data = data
        self.bins = bins

    def _plot(self):
        (bins_x, bins_y) = (self.bins, self.bins) if isinstance(self.bins, int) else self.bins
        X, Y = self.data
        H, ex, ey = numpy.histogram2d(X, Y, bins=(bins_x, bins_y))
        x_center = numpy.diff(ex) / 2 + ex[0:-1]
        x_digit = numpy.digitize(X, ex)
        y_center = numpy.empty(bins_y)
        y_std = numpy.empty(bins_y)
        for i in range(bins_x):
            y_pop = Y[numpy.where(x_digit == i + 1)[0]]
            y_center[i] = numpy.mean(y_pop)
            y_std[i] = numpy.std(y_pop)

        plt.hexbin(X, Y, bins='log')
        plt.errorbar(x_center, y_center, y_std, fmt='r-')
        cb = plt.colorbar()
        plt.xlim(ex[0], ex[-1])
        cb.set_label('log10(N)')

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for TMVA")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


"""
Helper module for displaying ROOT canvases in ipython notebooks

Usage example:

    canvas1 = default_canvas()
    sinc_function = TF1( 'fun1', 'abs(sin(x)/x)', 0, 10)
    canvas1.SetGridx()
    canvas1.SetGridy()
    sinc_function.Draw()
    canvas1

@author alexander.mazurov@cern.ch
@author andrey.ustyuzhanin@cern.ch
@date 2013-08-09
"""


def canvas(name="canvas1", size=(800, 600)):
    """
    Helper method for creating canvas
    If canvas with this name already exists, it will be returned
    """
    import ROOT
    # Check if canvas already exists
    canvas = ROOT.gROOT.FindObject(name)
    if canvas:
        return canvas
    else:
        width, height = size
        return ROOT.TCanvas(name, name, width, height)


def _display_canvas(canvas):
    with tempfile.NamedTemporaryFile(suffix=".png") as file_png:
        canvas.SaveAs(file_png.name)
        ip_img = display.Image(filename=file_png.name, format='png', embed=True)
    return ip_img._repr_png_()


try:
    import ROOT
    ROOT.gROOT.SetBatch()

    # register display function with PNG formatter:
    png_formatter = get_ipython().display_formatter.formatters['image/png']
    png_formatter.for_type(ROOT.TCanvas, _display_canvas)
except:
    pass
