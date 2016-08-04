"""
These classes are wrappers for physics machine learning library TMVA used .root format files (c++ library).
Now you can simply use it in python. TMVA contains classification and regression algorithms, including neural networks.
See `TMVA guide <http://mirror.yandex.ru/gentoo-distfiles/distfiles/TMVAUsersGuide-v4.03.pdf>`_
for the list of the available algorithms and parameters.
"""
from __future__ import division, print_function, absolute_import
from abc import ABCMeta
from logging import getLogger
import os
import tempfile
import subprocess
from subprocess import PIPE
import shutil
import sys

from .interface import Classifier, Regressor
from .utils import check_inputs, score_to_proba, proba_to_two_dimensions
from six.moves import cPickle
import signal

__author__ = 'Tatiana Likhomanenko, Alex Rogozhnikov'

logger = getLogger(__name__)
# those parameters that shall not be passed to the options of the TMVA estimators
_IGNORED_PARAMETERS = {'random_state'}
__all__ = ['TMVAClassifier', 'TMVARegressor']


class _AdditionalInformation:
    """
    Additional information for the tmva factory (used in training)
    """

    def __init__(self, directory, model_type='classification'):
        self.directory = directory
        self.tmva_root = 'result.root'
        self.tmva_job = "TMVAEstimation"
        self.model_type = model_type


class _AdditionalInformationPredict:
    """
    Additional information for the tmva factory (used to predict new data)
    """

    def __init__(self, directory, xml_file, method_name, model_type=('classification', None)):
        self.directory = directory
        self.xml_file = xml_file
        self.method_name = method_name
        self.model_type = model_type
        self.result_filename = os.path.join(directory, 'dump_predictions.pkl')


class TMVABase(object):
    """
    TMVABase is a base class for the tmva classification and regression models.

    :param str method: algorithm method (default='kBDT')
    :param features: features used in training
    :type features: list[str] or None
    :param str factory_options: system options, including data transformation before training
    :param dict method_parameters: estimator options

    .. note:: TMVA doesn't support staged predictions and features importances :(
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 factory_options="",
                 method='kBDT',
                 **method_parameters):
        self.method = method
        self._method_name = 'REP_Estimator'
        self.factory_options = factory_options
        self.method_parameters = method_parameters

        # contents of xml file with formula, read into memory
        self.formula_xml = None

    @staticmethod
    def _create_tmp_directory():
        return tempfile.mkdtemp(dir=os.getcwd())

    @staticmethod
    def _remove_tmp_directory(directory):
        shutil.rmtree(directory, ignore_errors=True)

    def _fit(self, X, y, sample_weight=None, model_type='classification'):
        """
        Train the estimator.

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param y: targets for samples --- array-like of shape [n_samples]
        :param sample_weight: weights for samples,
               array-like of shape [n_samples] or None if all weights are equal
        :return: self
        """
        # saving data to 2 different root files.
        directory = self._create_tmp_directory()
        add_info = _AdditionalInformation(directory, model_type=model_type)
        try:
            self._run_tmva_training(add_info, X, y, sample_weight)
        finally:
            self._remove_tmp_directory(directory)

        return self

    def _run_tmva_training(self, info, X, y, sample_weight):
        """
        Run subprocess to train tmva factory.

        :param info: class with additional information
        """
        tmva_process = None
        _platform = sys.platform
        try:
            if _platform == 'win32' or _platform == 'cygwin':
                tmva_process = subprocess.Popen(
                        '{executable} -c "import os; from rep.estimators import _tmvaFactory; _tmvaFactory.main()"'.format(
                                executable=sys.executable),
                        cwd=info.directory,
                        stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT)

            else:
                # Problem with Mac OS El Capitan which is not garanteed to set DYLD_LIBRARY_PATH.
                # This DYLD_LIBRARY_PATH can be used in root_numpy for dynamic loading ROOT libraries
                # https://github.com/rootpy/root_numpy/issues/227#issuecomment-165981891
                tmva_process = subprocess.Popen(
                        'export DYLD_LIBRARY_PATH={dyld}; cd "{directory}";'
                        '{executable} -c "import os; from rep.estimators import _tmvaFactory; _tmvaFactory.main()"'.format(
                                dyld=os.environ.get('DYLD_LIBRARY_PATH', ""),
                                directory=info.directory,
                                executable=sys.executable),
                        stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT,
                        shell=True, preexec_fn=os.setsid)

            try:
                cPickle.dump(self, tmva_process.stdin)
                cPickle.dump(info, tmva_process.stdin)
                cPickle.dump(X, tmva_process.stdin)
                cPickle.dump(y, tmva_process.stdin)
                cPickle.dump(sample_weight, tmva_process.stdin)
            except:
                # continuing, next we check the output of process
                pass
            stdout, stderr = tmva_process.communicate()
            assert tmva_process.returncode == 0, \
                'ERROR: TMVA process is incorrect finished \n LOG: %s \n %s' % (stderr, stdout)
            if stdout is not None:
                print('%s' % (stdout))

            xml_filename = os.path.join(info.directory, 'weights',
                                        '{job}_{name}.weights.xml'.format(job=info.tmva_job, name=self._method_name))
            with open(xml_filename, 'r') as xml_file:
                self.formula_xml = xml_file.read()
        finally:
            if tmva_process is not None:
                try:
                    if _platform == 'win32' or _platform == 'cygwin':
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(tmva_process.pid)])
                    else:
                        os.killpg(tmva_process.pid, signal.SIGTERM)
                except:
                    pass

    def _check_fitted(self):
        assert self.formula_xml is not None, "Classifier wasn't fitted, please call `fit` first"

    def _predict(self, X, model_type=('classification', None)):
        """
        Predict data

        :param pandas.DataFrame X: data shape [n_samples, n_features]
        :param model_type: (classification/regression, type of output transformation)
        :return: predicted values of shape [n_samples]
        """
        self._check_fitted()

        directory = self._create_tmp_directory()
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix='.xml', dir=directory, delete=True) as file_xml:
                file_xml.write(self.formula_xml)
                file_xml.flush()
                add_info = _AdditionalInformationPredict(directory, file_xml.name, self._method_name,
                                                         model_type=model_type)
                prediction = self._run_tmva_predict(add_info, X)
        finally:
            self._remove_tmp_directory(directory)

        return prediction

    @staticmethod
    def _run_tmva_predict(info, data):
        """
        Run subprocess to predict new data by tmva factory

        :param info: class with additional information
        """
        tmva_process = None
        _platform = sys.platform
        try:
            if _platform == 'win32' or _platform == 'cygwin':
                tmva_process = subprocess.Popen(
                        '{executable} -c "from rep.estimators import _tmvaReader; _tmvaReader.main()"'.format(
                                executable=sys.executable),
                        cwd=info.directory,
                        stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT)

            else:
                # Problem with Mac OS El Capitan (10.11) which is not guaranteed to set DYLD_LIBRARY_PATH.
                # This DYLD_LIBRARY_PATH can be used in root_numpy for dynamic loading ROOT libraries
                # https://github.com/rootpy/root_numpy/issues/227#issuecomment-165981891
                tmva_process = subprocess.Popen(
                        'export DYLD_LIBRARY_PATH={dyld}; cd "{directory}";'
                        '{executable} -c "from rep.estimators import _tmvaReader; _tmvaReader.main()"'.format(
                                dyld=os.environ.get('DYLD_LIBRARY_PATH', ""),
                                directory=info.directory,
                                executable=sys.executable),
                        stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT,
                        shell=True)

            try:
                cPickle.dump(info, tmva_process.stdin)
                cPickle.dump(data, tmva_process.stdin)
            except:
                # Doing nothing, there is check later.
                pass
            stdout, stderr = tmva_process.communicate()
            assert tmva_process.returncode == 0, \
                'ERROR: TMVA process is incorrect finished \n LOG: %s \n %s' % (stderr, stdout)
            with open(info.result_filename, 'rb') as predictions_file:
                predictions = cPickle.load(predictions_file)
            return predictions
        finally:
            if tmva_process is not None:
                try:
                    if _platform == 'win32' or _platform == 'cygwin':
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(tmva_process.pid)])
                    else:
                        os.killpg(tmva_process.pid, signal.SIGTERM)
                except:
                    pass


class TMVAClassifier(TMVABase, Classifier):
    """
    Implements classification models from TMVA library: CERN library for machine learning.

    :param str method: algorithm method (default='kBDT')
    :param features: features used in training
    :type features: list[str] or None
    :param str factory_options: system options, including data transformations before training, for example::

        "!V:!Silent:Color:Transformations=I;D;P;G,D"

    :param str sigmoid_function: function which is used to convert TMVA output to probabilities;

        * *identity* (use for svm, mlp) --- do not transform the output, use this value for methods returning class probabilities
        * *sigmoid* --- sigmoid transformation, use it if output varies in range [-infinity, +infinity]
        * *bdt* (for the BDT algorithms output varies in range [-1, 1])
        * *sig_eff=0.4* --- for the rectangular cut optimization methods,
          for instance, here 0.4 will be used as a signal efficiency to evaluate MVA,
          (put any float number from [0, 1])

    :param dict method_parameters: classifier options, example: `NTrees=100`, `BoostType='Grad'`.

    .. warning::
        TMVA doesn't support *staged_predict_proba()* and *feature_importances__*.

        TMVA doesn't support multiclassification, only two-class classification.

    `TMVA guide <http://mirror.yandex.ru/gentoo-distfiles/distfiles/TMVAUsersGuide-v4.03.pdf>`_.
    """
    def __init__(self,
                 method='kBDT',
                 features=None,
                 factory_options="",
                 sigmoid_function='bdt',
                 **method_parameters):
        TMVABase.__init__(self, factory_options=factory_options, method=method, **method_parameters)
        Classifier.__init__(self, features=features)
        self.sigmoid_function = sigmoid_function

    def _set_classes_special(self, y):
        self._set_classes(y)
        assert self.n_classes_ == 2, "Support only 2 classes (data contain {})".format(self.n_classes_)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in the model
        """
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                if k in _IGNORED_PARAMETERS:
                    continue
                self.method_parameters[k] = v

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :return: dict, parameter names mapped to their values.
        """
        parameters = self.method_parameters.copy()
        parameters['method'] = self.method
        parameters['factory_options'] = self.factory_options
        parameters['features'] = self.features
        return parameters

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)
        X = self._get_features(X).copy()
        self._set_classes_special(y)
        if self.n_classes_ == 2:
            self.factory_options = '{}:AnalysisType=Classification'.format(self.factory_options)
        else:
            self.factory_options = '{}:AnalysisType=Multiclass'.format(self.factory_options)

        return self._fit(X, y, sample_weight=sample_weight)

    fit.__doc__ = Classifier.fit.__doc__

    def predict_proba(self, X):
        X = self._get_features(X)
        prediction = self._predict(X, model_type=('classification', self.sigmoid_function))
        return self._convert_output(prediction)

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def _convert_output(self, prediction):
        """
        Convert the output to the probabilities for each class.

        :param array prediction: predictions which will be converted
        :return: probabilities
        """
        variants = {'bdt', 'sigmoid', 'identity'}
        if 'sig_eff' in self.sigmoid_function:
            return proba_to_two_dimensions(prediction)
        assert self.sigmoid_function in variants, \
            'sigmoid_function parameter must be one of {}, instead of {}'.format(variants, self.sigmoid_function)
        if self.sigmoid_function == 'sigmoid':
            return score_to_proba(prediction)
        elif self.sigmoid_function == 'bdt':
            return proba_to_two_dimensions((prediction + 1.) / 2.)
        else:
            return proba_to_two_dimensions(prediction)

    def staged_predict_proba(self, X):
        """
        .. warning:: This function is not supported for the TMVA library (**AttributeError** will be thrown)
        """
        raise AttributeError("'staged_predict_proba' is not supported by the TMVA library")


class TMVARegressor(TMVABase, Regressor):
    """
    Implements regression models from TMVA library: CERN library for machine learning.

    :param str method: algorithm method (default='kBDT')
    :param features: features used in training
    :type features: list[str] or None
    :param str factory_options: system options, including data transformations before training, for example::

        "!V:!Silent:Color:Transformations=I;D;P;G,D"

    :param dict method_parameters: regressor options, for example: `NTrees=100`, `BoostType='Grad'`

    .. warning::
        TMVA doesn't support *staged_predict()* and *feature_importances__*.

    `TMVA guide <http://mirror.yandex.ru/gentoo-distfiles/distfiles/TMVAUsersGuide-v4.03.pdf>`_
    """
    def __init__(self,
                 method='kBDT',
                 features=None,
                 factory_options="",
                 **method_parameters):
        TMVABase.__init__(self, factory_options=factory_options, method=method, **method_parameters)
        Regressor.__init__(self, features=features)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param dict params: parameters to set in the model
        """
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                if k in _IGNORED_PARAMETERS:
                    continue
                self.method_parameters[k] = v

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :return: dict, parameter names mapped to their values.
        """
        parameters = self.method_parameters.copy()
        parameters['method'] = self.method
        parameters['factory_options'] = self.factory_options
        parameters['features'] = self.features
        return parameters

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)
        X = self._get_features(X).copy()

        self.factory_options = '{}:AnalysisType=Regression'.format(self.factory_options)
        return self._fit(X, y, sample_weight=sample_weight, model_type='regression')

    fit.__doc__ = Regressor.fit.__doc__

    def predict(self, X):
        X = self._get_features(X)
        return self._predict(X, model_type=('regression', None))

    predict.__doc__ = Regressor.predict.__doc__

    def staged_predict(self, X):
        """
        .. warning:: This function is not supported for the TMVA library (**AttributeError** will be thrown)
        """
        raise AttributeError("'staged_predict' is not supported by the TMVA library")
