"""
:class:`MatrixNetClassifier` and :class:`MatrixNetRegressor` are wrappers for MatrixNet web service - proprietary BDT
developed at Yandex. Think about this as a specific Boosted Decision Tree algorithm which is available as a service.
At this moment MatrixMet is available only for **CERN users**.

To use MatrixNet, first acquire token::
 * Go to https://yandex-apps.cern.ch/ (login with your CERN-account)
 * Click `Add token` at the left panel
 * Choose service `MatrixNet` and click `Create token`
 * Create `~/.rep-matrixnet.config.json` file with the following content
   (custom path to the config file can be specified when creating a wrapper object)::

        {
            "url": "https://ml.cern.yandex.net/v1",

            "token": "<your_token>"
        }

"""

from __future__ import division, print_function, absolute_import

import contextlib
import hashlib
import json
import numbers
import os
import shutil
import tempfile
import time
from abc import ABCMeta
from collections import defaultdict
from copy import deepcopy
from logging import getLogger

import numpy
import pandas
from six import BytesIO
from sklearn.utils import check_random_state

from ._matrixnetapplier import MatrixNetApplier
from ._mnkit import MatrixNetClient
from .interface import Classifier, Regressor
from .utils import check_inputs, score_to_proba, remove_first_line
from ..utils import take_last

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

    :param features: features used in training
    :type features: list[str] or None
    :param str api_config_file: path to the file with remote api configuration in the json format::

                {"url": "https://ml.cern.yandex.net/v1", "token": "<your_token>"}

    :param int iterations: number of constructed trees (default=100)
    :param float regularization: regularization number (default=0.01)
    :param intervals: number of bins for features discretization or dict with borders
     list for each feature for its discretization (default=8)
    :type intervals: int or dict(str, list)
    :param int max_features_per_iteration: depth (default=6, supports 1 <= .. <= 6)
    :param float features_sample_rate_per_iteration: training features sampling (default=1.0)
    :param float training_fraction: training rows bagging (default=0.5)
    :param auto_stop: error value for training pre-stopping
    :type auto_stop: None or float
    :param bool sync: synchronous or asynchronous training on the server
    :param random_state: state for a pseudo random generator
    :type random_state: None or int or RandomState
    """
    __metaclass__ = ABCMeta
    _model_type = None

    def __init__(self, api_config_file=DEFAULT_CONFIG_PATH,
                 iterations=100, regularization=0.01, intervals=8,
                 max_features_per_iteration=6, features_sample_rate_per_iteration=1.0,
                 training_fraction=0.5, auto_stop=None, sync=True, random_state=42):

        self.api_config_file = api_config_file
        self.iterations = iterations
        self.regularization = regularization
        self.intervals = intervals
        self.auto_stop = auto_stop
        self.max_features_per_iteration = max_features_per_iteration
        self.features_sample_rate_per_iteration = features_sample_rate_per_iteration
        self.training_fraction = training_fraction
        self.sync = sync
        self.random_state = random_state
        self._initialisation_before_fit()

    def _initialisation_before_fit(self):
        self.formula_mx = None
        self.mn_cls = None

        self._api = None
        self._feature_importances = None
        self._pool_hash = None
        self._fit_status = False

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
        with open(filename, 'rb') as file_d:
            for chunk in iter(lambda: file_d.read(128 * md5.block_size), b''):
                md5.update(chunk)
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
            mn_bucket = self._api.bucket(bucket_id=self._pool_hash)
            if 'data.csv' not in set(mn_bucket.ls()):
                mn_bucket.upload(data_local)
        return mn_bucket

    def _train_formula(self, mn_bucket, features):
        """
        prepare parameters and call _train_sync
        """
        if self.random_state is None:
            seed = None
        elif isinstance(self.random_state, int):
            seed = self.random_state
        else:
            seed = check_random_state(self.random_state).randint(0, 10000)

        mn_options = {'iterations': int(self.iterations),
                      'regularization': float(self.regularization),
                      'max_features_per_iteration': int(self.max_features_per_iteration),
                      'features_sample_rate_per_iteration': float(self.features_sample_rate_per_iteration),
                      'training_fraction': float(self.training_fraction),
                      'seed': None,
                      'intervals': None,
                      'auto_stop': None,
                      'train_type': self._model_type}

        if seed is not None:
            mn_options['seed'] = int(seed)
        if isinstance(self.intervals, numbers.Number):
            mn_options['intervals'] = int(self.intervals)
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

            mn_options['intervals'] = 'borders' + suffix

        if self.auto_stop is not None:
            mn_options['auto_stop'] = float(self.auto_stop)

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
        print(self.mn_cls)
        assert self.mn_cls.get_status() != 'failed', 'Estimator is failed, run resubmit function, job id {}'.format(
            self.mn_cls.classifier_id)

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
            with open(outfile.name, 'rb') as formula_file:
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

    def __init__(self, features=None, api_config_file=DEFAULT_CONFIG_PATH,
                 iterations=100, regularization=0.01, intervals=8,
                 max_features_per_iteration=6, features_sample_rate_per_iteration=1.0,
                 training_fraction=0.5, auto_stop=None, sync=True, random_state=42):
        MatrixNetBase.__init__(self, api_config_file=api_config_file,
                               iterations=iterations, regularization=regularization, intervals=intervals,
                               max_features_per_iteration=max_features_per_iteration,
                               features_sample_rate_per_iteration=features_sample_rate_per_iteration,
                               training_fraction=training_fraction, auto_stop=auto_stop,
                               sync=sync, random_state=random_state)
        Classifier.__init__(self, features=features)
        self._model_type = 'classification'

    def _set_classes_special(self, y):
        self._set_classes(y)
        assert self.n_classes_ == 2, "Support only 2 classes (data contain {})".format(self.n_classes_)

    def fit(self, X, y, sample_weight=None):
        self._initialisation_before_fit()
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)

        self._set_classes_special(y)
        X = self._get_features(X)
        mn_bucket = self._upload_training_to_bucket(X, y, sample_weight)
        self._train_formula(mn_bucket, list(X.columns))

        if self.sync:
            self.synchronize()
        return self

    fit.__doc__ = Classifier.fit.__doc__

    def predict_proba(self, X):
        return take_last(self.staged_predict_proba(X, step=100000))

    predict_proba.__doc__ = Classifier.predict_proba.__doc__

    def staged_predict_proba(self, X, step=10):
        """
        Predict probabilities for data for each class label on each stage.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param int step: step for returned iterations (10 by default).

        :return: iterator
        """
        self.synchronize()

        X = self._get_features(X)

        data = X.astype(float)
        data = pandas.DataFrame(data)
        mx = MatrixNetApplier(BytesIO(self.formula_mx))
        for stage, prediction in enumerate(mx.staged_apply(data)):
            if stage % step == 0:
                yield score_to_proba(prediction)
        if stage % step != 0:
            yield score_to_proba(prediction)


class MatrixNetRegressor(MatrixNetBase, Regressor):
    __doc__ = 'MatrixNet for regression model. \n' + remove_first_line(MatrixNetBase.__doc__)

    def __init__(self, features=None, api_config_file=DEFAULT_CONFIG_PATH,
                 iterations=100, regularization=0.01, intervals=8,
                 max_features_per_iteration=6, features_sample_rate_per_iteration=1.0,
                 training_fraction=0.5, auto_stop=None, sync=True, random_state=42):
        MatrixNetBase.__init__(self, api_config_file=api_config_file,
                               iterations=iterations, regularization=regularization, intervals=intervals,
                               max_features_per_iteration=max_features_per_iteration,
                               features_sample_rate_per_iteration=features_sample_rate_per_iteration,
                               training_fraction=training_fraction, auto_stop=auto_stop, sync=sync,
                               random_state=random_state)
        Regressor.__init__(self, features=features)
        self._model_type = 'regression'

    def fit(self, X, y, sample_weight=None):
        self._initialisation_before_fit()
        X, y, sample_weight = check_inputs(X, y, sample_weight=sample_weight, allow_none_weights=False)

        X = self._get_features(X)
        mn_bucket = self._upload_training_to_bucket(X, y, sample_weight)
        self._train_formula(mn_bucket, list(X.columns))

        if self.sync:
            self.synchronize()
        return self

    fit.__doc__ = Classifier.fit.__doc__

    def predict(self, X):
        return take_last(self.staged_predict(X, step=100000))

    predict.__doc__ = Classifier.predict.__doc__

    def staged_predict(self, X, step=10):
        """
        Predict probabilities for data for each class label on each stage.

        :param pandas.DataFrame X: data of shape [n_samples, n_features]
        :param int step: step for returned iterations (10 by default).

        :return: iterator
        """
        self.synchronize()

        X = self._get_features(X)

        data = X.astype(float)
        data = pandas.DataFrame(data)
        mx = MatrixNetApplier(BytesIO(self.formula_mx))
        for stage, prediction in enumerate(mx.staged_apply(data)):
            if stage % step == 0:
                yield prediction
        if stage % step != 0:
            yield prediction
