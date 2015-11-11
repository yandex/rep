"""
There are different plotting backends supported:

    * *matplotlib* (default, de-facto standard plotting library),
    * *plotly* (proprietary package with interactive plots, information is kept on the server),
    * *ROOT* (the library used by CERN people),
    * *bokeh* (open-source package with interactive plots)
"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta, abstractmethod
import itertools
import os
from IPython import get_ipython

import matplotlib
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

matplotlib.rcParams['axes.color_cycle'] = COLOR_ARRAY

__author__ = 'Tatiana Likhomanenko'


class AbstractPlot(object):
    """
    Abstract class for possible plot objects, which implements plot function.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.PLOTLY_RESIZE = 60
        self.BOKEH_RESIZE = 50
        self.TMVA_RESIZE = 80
        self.xlim = None
        self.ylim = None
        self.xlabel = ""
        self.ylabel = ""
        self.title = ""
        self.figsize = (13, 7)
        self.fontsize = 14
        self.plotly_filename = 'example'
        self.new_plot = False
        self.canvas = None
        self._tmva_keeper = []

    @staticmethod
    def _plotly_config():
        try:
            import ConfigParser
        except ImportError:
            # python 3
            import configparser as ConfigParser

        config = ConfigParser.RawConfigParser()
        config.read('config_plotly')
        configParameters = config.defaults()
        api_user = configParameters['api_user'].strip()
        api_key = configParameters['api_key'].strip()
        user = configParameters['user'].strip()

        return api_user, api_key, user

    @abstractmethod
    def _plot(self):
        pass

    @abstractmethod
    def _plot_plotly(self, layout):
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

            %pylab inline

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
        """
        global _COLOR_CYCLE_BOKEH
        import bokeh.plotting as bkh
        from bokeh.models import Range1d
        from bokeh.properties import value

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

        bkh.output_notebook()

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

    def plot_plotly(self, plotly_filename=None, mpl_type=False, xlim=None, ylim=None, title=None, figsize=None,
                    xlabel=None, ylabel=None, fontsize=None, show_legend=True, grid=False):
        """
        Plot data using plotly library in IPython

        :param plotly_filename: name for resulting plot file on server (use unique name, else the same plot will be showen)
        :type plotly_filename: None or str
        :param bool mpl_type: use or not plotly converter from matplotlib (experimental parameter)
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
        import plotly.plotly as py
        from plotly import graph_objs
        from ipykernel import connect

        plotly_filename = self.plotly_filename if plotly_filename is None else plotly_filename
        try:
            connection_file_path = connect.find_connection_file()
            connection_file = os.path.basename(connection_file_path)
            if '-' in connection_file:
                kernel_id = connection_file.split('-', 1)[1].split('.')[0]
            else:
                kernel_id = connection_file.split('.')[0]
        except Exception as e:
            kernel_id = "no_kernel"

        PLOTLY_API_USER, PLOTLY_API_KEY, PLOTLY_USER = self._plotly_config()
        save_name = '{user}_{id}:{name}'.format(user=PLOTLY_USER, id=kernel_id, name=plotly_filename)
        py.sign_in(PLOTLY_API_USER, PLOTLY_API_KEY)

        if mpl_type:
            self.plot(new_plot=True, xlim=xlim, ylim=ylim, title=title, figsize=figsize, xlabel=xlabel, ylabel=ylabel,
                      fontsize=fontsize, grid=grid)
            mpl_fig = plt.gcf()
            update = dict(
                layout=dict(
                    showlegend=show_legend
                ),
                data=[dict(name=leg) for leg in mpl_fig.legends]
            )

            return py.iplot_mpl(mpl_fig, width=self.figsize[0] * 60,
                                update=update,
                                height=self.figsize[1] * 60,
                                filename=save_name,
                                fileopt='overwrite')

        xlabel = self.xlabel if xlabel is None else xlabel
        ylabel = self.ylabel if ylabel is None else ylabel
        title = self.title if title is None else title
        figsize = self.figsize if figsize is None else figsize
        fontsize = self.fontsize if fontsize is None else fontsize

        layout = graph_objs.Layout(yaxis={'title': ylabel, 'ticks': ''}, xaxis={'title': xlabel, 'ticks': ''},
                                   showlegend=show_legend, title=title,
                                   font=graph_objs.Font(family='Courier New, monospace', size=fontsize),
                                   width=figsize[0] * self.PLOTLY_RESIZE,
                                   height=figsize[1] * self.PLOTLY_RESIZE
        )

        fig = self._plot_plotly(layout)

        return py.iplot(fig, width=figsize[0] * self.PLOTLY_RESIZE, height=figsize[1] * self.PLOTLY_RESIZE,
                        filename=save_name)

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
            t = numpy.random.randint(10, 100000)
            figsize = (figsize[0] * self.TMVA_RESIZE, figsize[1] * self.TMVA_RESIZE)
            self.canvas = canvas("canvas%d" % t, figsize)

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
    """
    Implements grid plots

    Parameters:
    -----------
    :param int columns: count of columns in grid
    :param list[AbstractPlot] plots: plot objects
    """

    def __init__(self, columns=3, *plots):
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
        import bokeh.models as mdl
        import bokeh.plotting as bkh
        from bokeh.properties import value

        lst = []
        row_lst = []
        for plotter in self.plots:
            cur_plot = bkh.figure(title=plotter.title, plot_width=self.one_figsize[0] * self.BOKEH_RESIZE,
                                  plot_height=self.one_figsize[1] * self.BOKEH_RESIZE)
            if plotter.xlim is not None:
                cur_plot.x_range = mdl.Range1d(start=plotter.xlim[0], end=plotter.xlim[1])
            if plotter.ylim is not None:
                cur_plot.y_range = mdl.Range1d(start=plotter.ylim[0], end=plotter.ylim[1])
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
        grid = mdl.GridPlot(children=lst)
        return grid

    @staticmethod
    def _get_splts(n_row, n_col, n):

        n_splt = n_row * n_col
        n_empty = n_splt - n

        tmp1d = numpy.arange(1, n_splt + 1)  # => [1,2,..,n_splt]
        tmp2d = numpy.resize(tmp1d, (n_row, n_col))  # => [[1,2,..,N_rowcol],..[..,n_splt]]
        tmp2d_flip = tmp2d[::-1, :]  # => [[..,N_spl],..[1,2,..,N_rowcol]]

        tmp1d_in_order = tmp2d_flip.flatten().tolist()  # => [..,N_spl,..,1,2,..N_rowcol]

        splts_empty = range(n_col - n_empty + 1, n_col + 1)  # indices of empty subplots

        for splt in splts_empty:
            tmp1d_in_order.remove(splt)  # remove indices of empty subplots
        splts = tmp1d_in_order  # and get the complete list of subplots

        return splts, splts_empty

    def _plot_plotly(self, layout):
        import plotly.tools as tls
        from copy import deepcopy

        axis_style_empty = dict(
            title='',
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )

        fig = tls.make_subplots(rows=self.rows, cols=self.columns, horizontal_spacing=0.3 / self.columns,
                               vertical_spacing=0.3 / self.rows)
        splts, splts_empty = self._get_splts(self.rows, self.columns, len(self.plots))

        for index, plotter in zip(splts, self.plots):
            fig_one = plotter._plot_plotly(deepcopy(layout))
            for data in fig_one['data']:
                data['xaxis'] = 'x%d' % index
                data['yaxis'] = 'y%d' % index

            fig_one['layout']['xaxis%d' % index] = fig_one['layout']['xaxis']
            fig_one['layout']['yaxis%d' % index] = fig_one['layout']['yaxis']
            fig_one['layout']['xaxis%d' % index].update(anchor='y%d' % index,
                                                        title=plotter.xlabel + '<br>%s' % plotter.title)

            fig_one['layout']['yaxis%d' % index].update(title=plotter.ylabel)

            # fig_one['layout']['title%d' % index] = plotter.title

            fig_one['layout'].pop('xaxis')
            fig_one['layout'].pop('yaxis')
            fig_one['layout'].pop('title')

            fig['data'] += fig_one['data']

            fig['layout'].update(fig_one['layout'])

        for index in splts_empty:
            fig['layout']['xaxis{}'.format(index)].update(axis_style_empty)
            fig['layout']['yaxis{}'.format(index)].update(axis_style_empty)
        fig['layout'].update(autosize=False)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")


class HStackPlot(AbstractPlot):
    """
    Implements horizontal stack plots

    Parameters:
    -----------
    :param list[AbstractPlot] plots: plot objects
    """

    def __init__(self, *plots):
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

    def _plot_plotly(self, layout):
        obj = GridPlot(len(self.plots), *self.plots)
        return obj._plot_plotly(layout)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        obj = GridPlot(len(self.plots), *self.plots)
        return obj._plot_bokeh(current_plot, show_legend=show_legend)


class VStackPlot(AbstractPlot):
    """
    Implements vertical stack plots

    Parameters:
    -----------
    :param list[AbstractPlot] plots: plot objects
    """

    def __init__(self, *plots):
        super(VStackPlot, self).__init__()
        self.plots = plots

    def _plot(self):
        for i, plotter in enumerate(self.plots):
            plt.subplot(len(self.plots), 1, i + 1)
            plotter.plot(fontsize=self.fontsize_, show_legend=self.show_legend_)

    def _plot_plotly(self, layout):
        obj = GridPlot(1, *self.plots)
        return obj._plot_plotly(layout)

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        obj = GridPlot(1, *self.plots)
        return obj._plot_bokeh(current_plot, show_legend=show_legend)


class ErrorPlot(AbstractPlot):
    """
    Implements error bars plots

    Parameters:
    -----------
    :param errors: name - x points, y points, y errors, x errors
    :type errors: dict[str, tuple(array, array, array, array)]
    :param int size: size of scatters
    param bool log: logarithm scaling
    """

    def __init__(self, errors, size=2, log=False):
        super(ErrorPlot, self).__init__()
        self.errors = errors
        self.size = size
        self.log = log

    def _plot(self):
        for name, val in self.errors.items():
            x, y, yerr, xerr = val
            yerr_mod = yerr
            if self.log:
                y_mod = numpy.log(y)
                if yerr is not None:
                    yerr_mod = numpy.log(y + yerr) - y_mod
            else:
                y_mod = y
            err_bar = plt.errorbar(x, y_mod, yerr=yerr_mod, xerr=xerr, label=name, fmt='o', ms=self.size)
            err_bar[0].set_label('_nolegend_')

    def _plot_plotly(self, layout):
        data = []
        for name, val in self.errors.items():
            color = next(_COLOR_CYCLE)
            x, y, yerr, xerr = val
            yerr_mod = yerr
            if self.log:
                y_mod = numpy.log(y)
                if yerr is not None:
                    yerr_mod = numpy.log(y + yerr) - y_mod
            else:
                y_mod = y
            data.append({"x": x,
                         "y": y_mod,
                         'error_y': {'type': 'data',
                                     'array': yerr_mod,
                                     'thickness': 1,
                                     'width': 3,
                                     'opacity': 1,
                                     'color': color},
                         'error_x': {'type': 'data',
                                     'array': xerr,
                                     'thickness': 1,
                                     'width': 3,
                                     'opacity': 1,
                                     "color": color},
                         "name": name,
                         "type": "scatter",
                         'mode': 'markers',

                         "marker": {
                             "opacity": 30,
                             "size": self.size,
                             "color": color
                         }
            })
        from plotly import graph_objs

        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        import ROOT

        mg = ROOT.TMultiGraph()
        leg = ROOT.TLegend(0.2, 0.2, 0.5, 0.4)
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
            mg.Add(gr)
            leg.AddEntry(gr, name)
        mg.Draw("AP")
        return mg, leg

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class FunctionsPlot(AbstractPlot):
    """
    Implements 1d-function plots

    Parameters:
    -----------
    :param functions: dict which maps label of curve to x, y coordinates of points
    :type functions: dict[str, tuple(array, array)]
    """

    def __init__(self, functions):
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

    def _plot_plotly(self, layout):
        data = []

        for name, data_xy in self.functions.items():
            color = next(_COLOR_CYCLE)
            x_val, y_val = data_xy
            data.append({
                'name': name,
                'x': x_val,
                'y': y_val,
                'line': {'color': color}
            })
        from plotly import graph_objs

        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        import ROOT

        mg = ROOT.TMultiGraph()
        leg = ROOT.TLegend(0.2, 0.2, 0.5, 0.4)
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
            mg.Add(gr)
            leg.AddEntry(gr, name, "L")
        mg.Draw("AC")
        return mg, leg


class ColorMap(AbstractPlot):
    """
    Implements color map plots

    Parameters:
    -----------
    :param numpy.ndarray matrix: matrix
    :param labels: names for each matrix-row
    :type labels: None or list[str]
    :param str cmap: color map name
    :param float vmin: min value for color map
    :param float vmax: max value for color map

    .. note:: for plotly use

        * 'Greys', black to light-grey
        * 'YIGnBu', white to green to blue to dark-blue
        * 'Greens', dark-green to light-green
        * 'YIOrRd', red to orange to gold to tan to white
        * 'Bluered', bright-blue to purple to bright-red
        * 'RdBu', blue to red (dim, the default color scale)
        * 'Picnic', blue to light-blue to white to pink to red

        currently only available from the GUI, a slight alternative to 'Jet'

        * 'Portland', blue to green to yellow to orange to red (dim)
        * 'Jet', blue to light-blue to green to yellow to orange to red (bright)
        * 'Hot', tan to yellow to red to black
        * 'Blackbody', black to red to yellow to white to light-blue
        * 'Earth', blue to green to yellow to brown to tan to white
        * 'Electric', black to purple to orange to yellow to tan to white
    """

    def __init__(self, matrix, labels=None, cmap='jet', vmin=-1, vmax=1):
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

    def _plot_plotly(self, layout):
        from plotly import graph_objs

        colorbar_plotly = graph_objs.ColorBar(
            thickness=15,  # color bar thickness in px
            ticks='outside',  # tick outside colorbar
        )
        data = [{'type': 'heatmap',
                 'z': self.matrix,  # link 2D array
                 'x': self.labels,  # link x-axis labels
                 'y': self.labels,  # link y-axis labels
                 'colorscale': self.cmap,  # (!) select pre-defined colormap
                 'colorbar': colorbar_plotly,
                 'zmin': self.vmin,
                 'zmax': self.vmax,
                 'zauto': False
                }]
        layout['xaxis'].update(
            tickangle=-90
        )

        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

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
            hover = HoverTool(plot=current_plot, always_active=True)
        hover.tooltips = OrderedDict([
            ('labels', '@x @y'),
            ('value', '@value')
        ])
        current_plot.tools.append(hover)
        return current_plot


class ScatterPlot(AbstractPlot):
    """
    Implements scatters plots

    Parameters:
    -----------
    :param scatters: name - x points, y points
    :type scatters: dict[str, tuple(array, array)]
    :param int size: scatters size
    :param float alpha: transparency
    """

    def __init__(self, scatters, alpha=0.1, size=20):
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

    def _plot_plotly(self, layout):
        data = []
        for name, scatter in self.scatters.items():
            color = next(_COLOR_CYCLE)
            data.append({"x": scatter[0],
                         "y": scatter[1],
                         "name": name,
                         "type": "scatter",
                         'mode': 'markers',

                         "marker": {
                             "opacity": self.alpha,
                             "size": self.size // 5,
                             "color": color
                         }})
        from plotly import graph_objs

        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

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
    """
    Implements bar plots

    Parameters:
    -----------
    :param data: name - value, weight, style ('filled', another)
    :type data: dict[str, tuple(array, array, str)]
    :param bins: bins for histogram
    :type bins: int or list[float]
    :param bool normalization: normalize to pdf histogram or not
    :param value_range: min and max values
    :type value_range: None or tuple
    """

    def __init__(self, data, bins=30, normalization=True, value_range=None):
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

    def _plot_plotly(self, layout):
        from plotly import graph_objs

        data = []
        norm = "count"
        if self.normalization:
            norm = 'probability density'
        for label, sample in self.data.items():
            color = next(_COLOR_CYCLE)
            prediction, weight, style = sample
            if self.value_range is None:
                c_min, c_max = numpy.min(prediction), numpy.max(prediction)
            else:
                c_min, c_max = self.value_range
            data.append({
                'name': label,
                'x': prediction,
                'type': 'histogram',
                'histnorm': norm,
                'opacity': 0.5,
                'autobinx': False,
                'marker': {'color': color},
                'xbins': graph_objs.XBins(
                    start=c_min,
                    end=c_max,
                    size=1. * abs(c_max - c_min) / self.bins
                )
            })
        layout.update(barmode='overlay')
        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class BarComparePlot(AbstractPlot):
    """
    Implements bar plots

    Parameters:
    -----------
    :param data:
    :type data: dict[str, dict(str, float)]
    :param float alpha: opacity
    :param sortby: sort bars by this data key
    :type sortby: None or str
    :param int step: length
    """

    def __init__(self, data, alpha=0.5, sortby=None, step=5):
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

    def _plot_plotly(self, layout):
        from plotly import graph_objs

        if self.sortby is not None:
            inds = numpy.argsort(list(self.data[self.sortby].values()))[::-1]
        else:
            inds = numpy.arange(len(self.data[list(self.data.keys())[0]]))

        data = []
        for label, sample in self.data.items():
            color = next(_COLOR_CYCLE)
            data.append({
                'name': label,
                'x': numpy.array(list(sample.keys()))[inds],
                'y': numpy.array(list(sample.values()))[inds],
                'type': 'bar',
                'opacity': 0.5,
                'marker': {'color': color}
            })
        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        length = len(self.data) + self.step
        if self.sortby is not None:
            inds = numpy.argsort(list(self.data[self.sortby].values()))[::-1]
        else:
            inds = numpy.array(range(len(self.data[list(self.data.keys())[0]])))

        xticks_labels = numpy.array(list(self.data[list(self.data.keys())[0]].keys()))[inds]

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
    """
    Implements 2d-functions plots

    Parameters:
    -----------
    :param function function: vector function (X, Y)
    :param tuple(float, float) xlim: x ranges
    :param tuple(float, float) ylim: y ranges
    :param int xsteps: count of points for approximation on x-axis
    :param int ysteps: count of points for approximation on y-axis
    :param str cmap: color map
    :param float vmin: value, corresponding to minimum on cmap
    :param float vmax: value, corresponding to maximum on cmap

    .. note:: for plotly use

        * 'Greys', black to light-grey
        * 'YIGnBu', white to green to blue to dark-blue
        * 'Greens', dark-green to light-green
        * 'YIOrRd', red to orange to gold to tan to white
        * 'Bluered', bright-blue to purple to bright-red
        * 'RdBu', blue to red (dim, the default color scale)
        * 'Picnic', blue to light-blue to white to pink to red

        currently only available from the GUI, a slight alternative to 'Jet'

        * 'Portland', blue to green to yellow to orange to red (dim)
        * 'Jet', blue to light-blue to green to yellow to orange to red (bright)
        * 'Hot', tan to yellow to red to black
        * 'Blackbody', black to red to yellow to white to light-blue
        * 'Earth', blue to green to yellow to brown to tan to white
        * 'Electric', black to purple to orange to yellow to tan to white
    """

    def __init__(self, function, xlim, ylim, xsteps=100, ysteps=100, cmap='Blues',
                 vmin=None, vmax=None):
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

    def _plot_plotly(self, layout):
        from plotly import graph_objs

        colorbar_plotly = graph_objs.ColorBar(
            thickness=15,  # color bar thickness in px
            ticks='outside',  # tick outside colorbar
            title='value'
        )
        data = [{'type': 'heatmap',
                 'z': self.z,
                 'y': map(str, list(self.y[:, 0])),
                 'x': map(str, list(self.x[0, :])),
                 'colorscale': self.cmap,
                 'colorbar': colorbar_plotly,
                }]
        if self.vmin is not None:
            data[0]['zmin'] = self.vmin
            data[0]['zauto'] = False
        if self.vmax is not None:
            data[0]['zmax'] = self.vmax
            data[0]['zauto'] = False

        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported for bokeh")


class Histogram2D_Plot(AbstractPlot):
    """
    Implements correlations plots

    Parameters:
    -----------
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

    def __init__(self, data, bins=30, cmap='Blues', cmin=None, cmax=None, range=None, normed=False):
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

    def _plot_plotly(self, layout):
        from plotly import graph_objs

        colorbar_plotly = graph_objs.ColorBar(
            thickness=15,  # color bar thickness in px
            ticks='outside',  # tick outside colorbar
            title='value'
        )
        X, Y = self.data
        data = [{'type': 'histogram2d',
                 'y': Y,
                 'x': X,
                 'colorscale': self.cmap,
                 'colorbar': colorbar_plotly,
                }]
        if self.vmin is not None:
            data[0]['zmin'] = self.vmin
            data[0]['zauto'] = False
        if self.vmax is not None:
            data[0]['zmax'] = self.vmax
            data[0]['zauto'] = False
        if self.range is None:
            data[0]['nbinsx'] = self.binsX
            data[0]['nbinsy'] = self.binsY
        else:
            start, end = self.range[1]
            size = 1. * (end - start) / self.binsY
            data[0]['ybins']= {'start': start, 'end': end, 'size': size}
            data[0]['autobiny'] =False
            start, end = self.range[0]
            size = 1. * (end - start) / self.binsX
            data[0]['xbins']= {'start': start, 'end': end, 'size': size}
            data[0]['autobinx'] =False
        if self.normed:
            data[0]['histnorm'] = 'probability'


        fig = graph_objs.Figure(data=graph_objs.Data(data), layout=layout)
        return fig

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class CorrelationPlot(AbstractPlot):
    """
    Implements correlations plots

    Parameters:
    -----------
    :param (array, array) data: values for first, values for second
    :param bins: count of bins
    :type bins: int or list[float]
    """

    def __init__(self, data, bins=30):
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

    def _plot_plotly(self, layout):
        raise NotImplementedError("Not supported for plotly")

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


class CorrelationMapPlot(AbstractPlot):
    """
    Implements correlations map plots

    Parameters:
    -----------
    :param (array, array) data: name var, name var - values for first, values for second
    :param bins: count of bins
    :type bins: int or list[float]
    """

    def __init__(self, data, bins=30):
        super(CorrelationMapPlot, self).__init__()
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

        plt.hexbin(X, Y, bins='log')
        plt.errorbar(x_center, y_center, y_std, fmt='r-')
        cb = plt.colorbar()
        plt.xlim(ex[0], ex[-1])
        cb.set_label('log10(N)')

    def _plot_plotly(self, layout):
        raise NotImplementedError("Not supported for plotly")

    def _plot_tmva(self):
        raise NotImplementedError("Not supported for tmva")

    def _plot_bokeh(self, current_plot, show_legend=True):
        raise NotImplementedError("Not supported by bokeh")


"""
Helper module for displaying ROOT canvases in ipython notebooks

Usage example:
    # Save this file as rootnotes.py to your working directory.

    c1 = default_canvas()
    fun1 = TF1( 'fun1', 'abs(sin(x)/x)', 0, 10)
    c1.SetGridx()
    c1.SetGridy()
    fun1.Draw()
    c1

More examples: http://mazurov.github.io/webfest2013/

@author alexander.mazurov@cern.ch
@author andrey.ustyuzhanin@cern.ch
@date 2013-08-09
"""


def canvas(name="icanvas", size=(800, 600)):
    """Helper method for creating canvas"""

    # Check if icanvas already exists
    canvas = ROOT.gROOT.FindObject(name)
    assert len(size) == 2
    if canvas:
        return canvas
    else:
        return ROOT.TCanvas(name, name, size[0], size[1])


def default_canvas(name="icanvas", size=(800, 600)):
    """ deprecated """
    return canvas(name=name, size=size)


def _display_canvas(canvas):
    with tempfile.NamedTemporaryFile(suffix=".png") as file_png:
        canvas.SaveAs(file_png.name)
        ip_img = display.Image(filename=file_png.name, format='png', embed=True)
    return ip_img._repr_png_()


def _display_any(obj):
    with tempfile.NamedTemporaryFile(suffix=".png") as file_png:
        obj.Draw()
        ROOT.gPad.SaveAs(file_png.name)
        ip_img = display.Image(filename=file_png.name, format='png', embed=True)
    return ip_img._repr_png_()

try:
    import ROOT
    ROOT.gROOT.SetBatch()

    # register display function with PNG formatter:
    png_formatter = get_ipython().display_formatter.formatters['image/png']  # noqa

    # Register ROOT types in ipython
    #
    #   In  [1]: canvas = rootnotes.canvas()
    #   In  [2]: canvas
    #   Out [2]: [image will be here]
    png_formatter.for_type(ROOT.TCanvas, _display_canvas)
    png_formatter.for_type(ROOT.TF1, _display_any)
except:
    pass

