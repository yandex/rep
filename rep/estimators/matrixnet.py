"""
:class:`MatrixNetClassifier` and :class:`MatrixNetRegressor` are wrappers for MatrixNet web service - proprietary BDT
developed at Yandex. Think about this as a specific Boosted Decision Tree algorithm which is available as a service.
At this moment MatrixMet is available only for **CERN users**.

To get the access to MatrixNet, you'll need:
 * Go to https://yandex-apps.cern.ch/
 * Login with your CERN-account
 * Click `Add token` at the left panel
 * Choose service `MatrixNet` and click `Create token`
 * Create `~/.rep-matrixnet.config.json` file with the following content (the path to config file can be changed in the constructor of the wrappers)::

        {
            "url": "https://ml.cern.yandex.net/v1",

            "token": "<your_token>"
        }

"""

from __future__ import division, print_function, absolute_import
from collections import defaultdict
import json
import numbers
from logging import getLogger
import hashlib
import tempfile
import time
import os
import shutil
import contextlib
from abc import ABCMeta

import pandas
import numpy
from six import StringIO
from sklearn.utils import check_random_state

from .interface import Classifier, Regressor
from .utils import check_inputs, score_to_proba, remove_first_line, _get_features

from ._matrixnetapplier import MatrixNetApplier
from copy import deepcopy

from ._mnkit import MatrixNetClient

__author__ = 'Tatiana Likhomanenko, Alex Rogozhnikov'
__all__ = ['MatrixNetBase', 'MatrixNetClassifier', 'MatrixNetRegressor']

logger = getLogger(__name__)

CHUNKSIZE = 1000
SYNC_SLEEP_TIME = 10
DEFAULT_CONFIG_PATH = "$HOME/.rep-matrixnet.config.json"


@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class MatrixNetBase(object):
    """Base class for MatrixNetClassifier and MatrixNetRegressor.

    This is a wrapper around **MatrixNet (specific BDT)** technology developed at **Yandex**,
    which is available for CERN people using authorization.
    Trained estimator is downloaded and stored at your computer, so you can use it at any time.

    :param train_features: features used in training
    :type train_features: list[str] or None
    :param baseline_feature: feature values of which are used as initial predictions
    :type baseline_feature: str or None'

    :param api_config_file: path to the file with remote api configuration in the json format::

                {"url": "https://ml.cern.yandex.net/v1", "token": "<your_token>"}

    :type api_config_file: str

    :param int iterations: number of constructed trees (default=100)
    :param float regularization: regularization number (default=0.01)
    :param intervals: number of bins for features discretization or dict with borders
     list for each feature for its discretisation (default=8)
    :type intervals: int or dict(str, list)
    :param int max_features_per_iteration: depth (default=6, supports 1 <= .. <= 6)
    :param float features_sample_rate_per_iteration: training features sampling (default=1.0)
    :param float training_fraction: training rows bagging (default=0.5)
    :param auto_stop: error value for training prestopping
    :type auto_stop: None or float
    :param str command_line_params: command line additional parameters for MatrixNet.
    :param bool sync: synchronic or asynchronic training on the server
    :param random_state: state for a pseudo random generator
    :type random_state: None or int or RandomState
    """
    __metaclass__ = ABCMeta
    _model_type = None

    def __init__(self, api_config_file=DEFAULT_CONFIG_PATH,
                 train_features=None, baseline_feature=None,
                 iterations=100, regularization=0.01, intervals=8,
                 max_features_per_iteration=6, features_sample_rate_per_iteration=1.0,
                 training_fraction=0.5, auto_stop=None, command_line_params=None, sync=True,
                 random_state=42):

        self.api_config_file = api_config_file
        self.iterations = iterations
        self.regularization = regularization
        self.intervals = intervals
        self.auto_stop = auto_stop
        self.train_features = train_features
        self.baseline_feature = baseline_feature
        self.max_features_per_iteration = max_features_per_iteration
        self.features_sample_rate_per_iteration = features_sample_rate_per_iteration
        self.training_fraction = training_fraction
        self.command_line_params = command_line_params
        self.sync = sync
        self.random_state = random_state
        self._train_type_options = ""
        self._initialisation_before_fit()

    def _initialisation_before_fit(self):
        self.formula_mx = None
        self.mn_cls = None

        self._api = None
        self._feature_importances = None
        self._pool_hash = None
        self._fit_status = False

    def _features(self):
        if self.train_features is None:
            return None
        else:
            if self.baseline_feature is not None and self.baseline_feature not in set(self.train_features):
                    return list(self.train_features) + [self.baseline_feature]
            else:
                return self.train_features

    def _get_features(self, X, allow_nans=False):
        """
        :param pandas.DataFrame X: train dataset

        :return: pandas.DataFrame with used features
        """
        baseline_column_values = None
        if self.baseline_feature is not None:
            baseline_column_values, _ = _get_features([self.baseline_feature], X, allow_nans=allow_nans)
            baseline_column_values = numpy.ravel(numpy.array(baseline_column_values))
        X_prepared, self.train_features = _get_features(self.train_features, X, allow_nans=allow_nans)
        self.features = self._features()
        return baseline_column_values, X_prepared

    def _configure_api(self, config_file_path):
        config_file_path = os.path.expandvars(config_file_path)
        with open(config_file_path, 'r') as conf_file:
            config = json.load(conf_file)
            self._api = MatrixNetClient(config['url'], token=config['token'])
            if self.mn_cls is not None:
                self.mn_cls.requests_kwargs['headers']['X-Yacern-Token'] = self._api.auth_token

    def __getstate__(self):
        result = deepcopy(self.__dict__)
        if '_api' in result:
            del result['_api']
        if result['mn_cls'] is not None:
            result['mn_cls'].requests_kwargs['headers']['X-Yacern-Token'] = ""
        return result

    def __convert_borders(self, borders, features):
        """
        convert borders for features into correct format to send to the server
        """
        converted_borders = ""
        for i, feature in enumerate(features):
            if not feature in borders:
                continue
            for border in borders[feature]:
                converted_borders += "{}\t{}\t0\n".format(i, border)
        return converted_borders

    def _md5(self, filename):
        """
        compute md5 hash for file
        """
        md5 = hashlib.md5()
        with open(filename, 'r') as file_d:
            for chunk in iter(lambda: file_d.read(128 * md5.block_size), b''):
                md5.update(chunk.encode('utf-8'))
        return md5.hexdigest()

    def _save_df_to_file(self, df, labels, sample_weight, outfile):
        """
        save DataFrame to send to server
        """
        header = True
        mode = 'w'
        for row in range(0, len(df), CHUNKSIZE):
            df_ef = df.iloc[row: row + CHUNKSIZE, :].copy()
            df_ef['is_signal'] = labels[row: row + CHUNKSIZE]
            df_ef['weight'] = sample_weight[row: row + CHUNKSIZE]
            df_ef.to_csv(outfile, sep='\t', index=False, header=header, mode=mode)
            header = False
            mode = 'a'

    def _upload_training_to_bucket(self, X, y, sample_weight):
        with make_temp_directory() as temp_dir:
            data_local = os.path.join(temp_dir, 'data.csv')
            self._save_df_to_file(X, y, sample_weight, data_local)
            self._pool_hash = self._md5(data_local)

            self._configure_api(self.api_config_file)
        #     mn_bucket = self._api.bucket(bucket_id=self._pool_hash)
        #     if 'data.csv' not in set(mn_bucket.ls()):
        #         mn_bucket.upload(data_local)
        # return mn_bucket

    def _train_formula(self, mn_bucket, features, baseline=None):
        """
        prepare parameters and call _train_sync
        """
        if self.random_state is None:
            seed = None
        elif isinstance(self.random_state, int):
            seed = self.random_state
        else:
            seed = check_random_state(self.random_state).randint(0, 10000)

        mn_options = '-i {iterations} -w {regularization} ' \
                     '-W -n {max_features_per_iteration} -Z {features_sample_rate_per_iteration} ' \
                     '-S {training_fraction}'

        mn_options = mn_options.format(
                iterations=int(self.iterations),
                regularization=self.regularization,
                max_features_per_iteration=int(self.max_features_per_iteration),
                training_fraction=self.training_fraction,
                features_sample_rate_per_iteration=self.features_sample_rate_per_iteration)

        if seed is not None:
            mn_options = "{params} -r {seed}".format(params=mn_options, seed=seed)
        if isinstance(self.intervals, numbers.Number):
            mn_options = "{params} -x {intervals}".format(params=mn_options, intervals=self.intervals)
        else:
            assert set(self.intervals.keys()) == set(features), 'intervals must contains borders for all features'
            with make_temp_directory() as temp_dir:
                borders_local = os.path.join(temp_dir, 'borders')
                with open(borders_local, "w") as file_b:
                    file_b.write(self.__convert_borders(self.intervals, features))

                suffix = '.{}.baseline'.format(self._md5(borders_local))
                borders_name = borders_local + suffix
                os.rename(borders_local, borders_name)
                if borders_name not in set(mn_bucket.ls()):
                    mn_bucket.upload(borders_name)

            mn_options = "{params} -B {name}".format(params=mn_options, name='borders' + suffix)

        if baseline is not None:
            with make_temp_directory() as temp_dir:
                baseline_path = os.path.join(temp_dir, 'train.txt')
                pandas.DataFrame({'baseline': baseline}).to_csv(baseline_path, index=False, header=False)
                suffix = '.{}.baseline'.format(self._md5(baseline_path))
                baseline_name = baseline_path + suffix
                os.rename(baseline_path, baseline_name)
                if baseline_name not in set(mn_bucket.ls()):
                    mn_bucket.upload(baseline_name)

            mn_options = "{params} -b {suffix}".format(params=mn_options, suffix=suffix)

        if self.auto_stop is not None:
            mn_options = "{params} {auto_stop}".format(params=mn_options,
                                                       auto_stop='--auto-stop %f' % self.auto_stop)

        if self.command_line_params is not None:
            mn_options = "{params} {cmd_line}".format(params=mn_options, cmd_line=self.command_line_params)
        if self._train_type_options is not None:
            mn_options = "{params} {type}".format(params=mn_options, type=self._train_type_options)

        descriptor = {
            'mn_parameters': mn_options,
            'mn_version': 1,
            'fields': list(features),
            'extra': {},
        }
        self._configure_api(self.api_config_file)
        self.mn_cls = self._api.classifier(
                parameters=descriptor,
                description="REP-submitted classifier",
                bucket_id=mn_bucket.bucket_id,
        )
        self.mn_cls.upload()

        self._fit_status = True

    def training_status(self):
        """
        Check if training has finished on the server

        :rtype: bool
        """
        self._configure_api(self.api_config_file)
        assert self._fit_status and self.mn_cls is not None, 'Call fit before'
        assert self.mn_cls.get_status() != 'failed', 'Estimator is failed, run resubmit function, job id {}'.format(
                self.mn_cls.cl_id)

        if self.mn_cls.get_status() == 'completed':
            self._download_formula()
            self._download_features()
            return True
        else:
            return False

    def synchronize(self):
        """
        Synchronise asynchronic training: wait while training process will be finished on the server
        """
        assert self._fit_status, 'Do fit, model is not trained'
        if self.formula_mx is not None and self._feature_importances is not None:
            return
        while not self.training_status():
            time.sleep(SYNC_SLEEP_TIME)
        assert (self.formula_mx is not None and self._feature_importances is not None), \
            "Classifier wasn't fitted, please call `fit` first"

    def _download_formula(self):
        """
        Download formula from the server
        """
        if self.formula_mx is not None:
            return
        with tempfile.NamedTemporaryFile() as outfile:
            self._configure_api(self.api_config_file)
            self.mn_cls.save_formula(outfile.name)
            with open(outfile.name, 'r') as formula_file:
                self.formula_mx = formula_file.read()
                assert len(self.formula_mx) > 0, "Formula is empty"

    def _download_features(self):
        if self._feature_importances is not None:
            return
        with tempfile.NamedTemporaryFile() as outfile:
            self.mn_cls.save_stats(outfile.name)
            stats = json.loads(open(outfile.name).read())['factors']
            importances = defaultdict(list)
            columns = ["name", "effect", "info", "efficiency"]
            for data in stats:
                for key in columns:
                    importances[key].append(data[key])

            df = pandas.DataFrame(importances)
            df_result = {'effect': df['effect'].values / max(df['effect']),
                         'information': df['info'].values / max(df['info']),
                         'efficiency': df['efficiency'].values / max(df['efficiency'])}
        self._feature_importances = pandas.DataFrame(df_result, index=df['name'].values)

    def get_feature_importances(self):
        """
        Get features importance: `effect`, `efficiency`, `information` characteristics

        :rtype: pandas.DataFrame with `index=self.features`
        """
        self.synchronize()
        return self._feature_importances

    @property
    def feature_importances_(self):
        """Sklearn-way of returning feature importance.
        This returned as numpy.array, 'effect' column is used among MatrixNet importances.
        """
        return numpy.array(self.get_feature_importances()['effect'].ix[self.features])

    @property
    def get_iterations(self):
        """
        Return number of already constructed trees during training

        :return: int or None
        """
        self._configure_api(self.api_config_file)
        if self.mn_cls is not None:
            return self.mn_cls.get_iterations()
        else:
            return None

    def resubmit(self):
        """
        Resubmit training process on the server in case of failing job.
        """
        if self.mn_cls is not None:
            self._configure_api(self.api_config_file)
            self.mn_cls.resubmit()


class MatrixNetClassifier(MatrixNetBase, Classifier):
    __doc__ = 'MatrixNet classification model. \n' + remove_first_line(MatrixNetBase.__doc__)

    def _set_classes_special(self, y):
        indices = self._set_classes(y)
        assert self.n_classes_ == 2, "Support only 2 classes (data contain {})".format(self.n_classes_)
        # if self.n_classes_ > 2:
        #     self.classes_mn_ = self.classes_[numpy.argsort(indices)]
        self.classes_mn_ = self.classes_

    def fit(self, X, y, sample_weight=None):
        self._initialisation_before_fit()
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)

        self._set_classes_special(y)
        if self.n_classes_ == 2:
            self._train_type_options = '-c --c-fast'
        else:
            assert self.baseline_feature is None, 'Baseline option is supported only for binary classification'
            self._train_type_options = '-m'
        baseline, X = self._get_features(X)
        mn_bucket = self._upload_training_to_bucket(X, y, sample_weight)
        # self._train_formula(mn_bucket, list(X.columns), baseline)
        #
        # if self.sync:
        #     self.synchronize()
        # return self

    fit.__doc__ = Classifier.fit.__doc__

    def predict_proba(self, X):
        self.synchronize()

        baseline, X = self._get_features(X)

        data = X.astype(float)
        data = pandas.DataFrame(data)
        mx = MatrixNetApplier(StringIO(self.formula_mx))

        if self.n_classes_ == 2:
            if baseline is None:
                return score_to_proba(mx.apply(data))
            else:
                return score_to_proba(baseline + mx.apply(data))
        else:
            return mx.apply(data)[:, numpy.argsort(self.classes_mn_)]

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X, step=10):
        """
        Predict probabilities for data for each class label on each stage..

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param int step: step for returned iterations (10 by default).

        :return: iterator
        """
        self.synchronize()

        baseline, X = self._get_features(X)

        data = X.astype(float)
        data = pandas.DataFrame(data)
        mx = MatrixNetApplier(StringIO(self.formula_mx))
        prediction = numpy.zeros(len(data))

        for stage, prediction_iteration in enumerate(mx.apply_separately(data)):
            prediction += prediction_iteration
            if stage % step == 0 or stage == self.iterations:
                if self.n_classes_ == 2:
                    if baseline is None:
                        yield score_to_proba(prediction)
                    else:
                        yield score_to_proba(baseline + prediction)
                else:
                    yield prediction[:, numpy.argsort(self.classes_mn_)]


class MatrixNetRegressor(MatrixNetBase, Regressor):
    __doc__ = 'MatrixNet for regression model. \n' + remove_first_line(MatrixNetBase.__doc__)
    _model_type = 'regression'

    def fit(self, X, y, sample_weight=None):
        self._initialisation_before_fit()
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)

        baseline, X = self._get_features(X)
        self._train_type_options = '--quad-fast'
        mn_bucket = self._upload_training_to_bucket(X, y, sample_weight)
        self._train_formula(mn_bucket, list(X.columns), baseline)

        if self.sync:
            self.synchronize()
        return self

    fit.__doc__ = Classifier.fit.__doc__

    def predict(self, X):
        self.synchronize()

        baseline, X = self._get_features(X)
        data = X.astype(float)
        data = pandas.DataFrame(data)
        mx = MatrixNetApplier(StringIO(self.formula_mx))
        if baseline is None:
            return mx.apply(data)
        else:
            return baseline + mx.apply(data)

    predict.__doc__ = Classifier.predict.__doc__

    def staged_predict(self, X, step=10):
        """
        Predict probabilities for data for each class label on each stage..

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param int step: step for returned iterations (10 by default).

        :return: iterator
        """
        self.synchronize()

        baseline, X = self._get_features(X)

        data = X.astype(float)
        data = pandas.DataFrame(data)
        mx = MatrixNetApplier(StringIO(self.formula_mx))
        prediction = numpy.zeros(len(data))

        for stage, prediction_iteration in enumerate(mx.apply_separately(data)):
            prediction += prediction_iteration
            if stage % step == 0 or stage == self.iterations:
                if baseline is None:
                    yield prediction
                else:
                    yield baseline + prediction
