from __future__ import division, print_function, absolute_import

import os
from time import time, sleep
from tempfile import mkstemp
from nose.tools import raises
import unittest
import json
from rep.estimators._mnkit import MatrixNetClient
from rep_ef.estimators import MatrixNetClassifier, MatrixNetRegressor
from rep.test.test_estimators import generate_classification_data, generate_regression_data

__author__ = 'Alexander Baranov, Tatiana Likhomanenko'


# test api errors
@raises(Exception)
def test_Exception_credential():
    X, y, sample_weight = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.fit(X, y, sample_weight=sample_weight)


@raises(Exception)
def test_Exception_server():
    X, y, sample_weight = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.fit(X, y, sample_weight=sample_weight)


@raises(AssertionError)
def test_Exception_predict_proba():
    X, _, _ = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.predict_proba(X)


@raises(AssertionError)
def test_Exception_staged_predict_proba():
    X, _, _ = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    for _ in cl.staged_predict_proba(X):
        pass


@raises(AssertionError)
def test_Exception_feature_importances():
    X, _, _ = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    print(cl.feature_importances_)


@raises(AssertionError)
def test_Exception_trained_status():
    X, _, _ = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.training_status()


@raises(AssertionError)
def test_Exception_synchronized():
    X, _, _ = generate_classification_data()
    cl = MatrixNetClassifier(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.synchronize()


@raises(AssertionError)
def test_Exception_reg_predict():
    X, _, _ = generate_regression_data()
    cl = MatrixNetRegressor(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.predict(X)


@raises(AssertionError)
def test_Exception_reg_staged_predict():
    X, _, _ = generate_regression_data()
    cl = MatrixNetRegressor(api_config_file='help_files/wrong_config.json', iterations=50)
    for _ in cl.staged_predict(X):
        pass


@raises(AssertionError)
def test_Exception_reg_feature_importances():
    X, _, _ = generate_regression_data()
    cl = MatrixNetRegressor(api_config_file='help_files/wrong_config.json', iterations=50)
    print(cl.feature_importances_)


@raises(AssertionError)
def test_Exception_reg_trained_status():
    X, _, _ = generate_regression_data()
    cl = MatrixNetRegressor(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.training_status()


@raises(AssertionError)
def test_Exception_reg_synchronized():
    X, _, _ = generate_regression_data()
    cl = MatrixNetRegressor(api_config_file='help_files/wrong_config.json', iterations=50)
    cl.synchronize()


class MatrixNetTest(unittest.TestCase):
    DEFAULT_CONFIG_PATH = "$HOME/.rep-matrixnet.config.json"

    def setUp(self):
        config_file_path = os.path.expandvars(self.DEFAULT_CONFIG_PATH)
        with open(config_file_path, 'r') as conf_file:
            config = json.load(conf_file)
        self.api_url = config['url']
        self.mn = MatrixNetClient(self.api_url, config['token'])


# test Bucket

class TestBuckets(MatrixNetTest):
    def test_create_delete(self):
        b1 = self.mn.bucket()
        b1.remove()

    def test_create_with_id(self):
        bucket_id = "testbucket" + str(int(time()))
        b1 = self.mn.bucket(bucket_id=bucket_id)
        b1.remove()

    def test_bucket_id(self):
        b1 = self.mn.bucket()
        b2 = self.mn.bucket(bucket_id=b1.bucket_id)
        b1.remove()

    def test_upload(self):
        b1 = self.mn.bucket()

        here = os.path.dirname(os.path.realpath(__file__))
        datapath = os.path.join(here, "help_files/data.csv")

        result = b1.upload(datapath)
        self.assertTrue(result)

        self.assertEqual(b1.ls(), [u'data.csv'])

        b1.remove()


# test Classifier

TEST_PARAMS = {
    'mn_parameters': '-i 10 -w 0.01 -x 8 -C 0.5 -W',
    'fields': [
        'FlightDistance',
        'FlightDistanceError',
        'IP',
        'IPSig',
        'VertexChi2',
        'weight'
    ],
    'extra': {
    },
}

# for some reason the task is pending all time.

# class TestEstimator(MatrixNetTest):
#     def test_classifier(self):
#         bucket_test = self.mn.bucket()
#
#         here = os.path.dirname(os.path.realpath(__file__))
#         datapath = os.path.join(here, "help_files/data.csv")
#
#         result = bucket_test.upload(datapath)
#         self.assertTrue(result)
#
#         cls = self.mn.classifier(
#                 parameters=TEST_PARAMS,
#                 description="Some description",
#                 bucket_id=bucket_test.bucket_id,
#         )
#         cls.upload()
#         status = cls.get_status()
#         while status != "completed":
#             status = cls.get_status()
#             iterations = cls.get_iterations()
#             print("Training: status={} iterations={}".format(status, iterations))
#             sleep(2)
#         print('finish training')
#         formula_tmp_local = mkstemp(dir='/tmp')[1]
#         cls.save_formula(formula_tmp_local)
#         os.remove(formula_tmp_local)
#
#         self.assertTrue(cls.resubmit())
#         status = cls.get_status()
#         while status != "completed":
#             status = cls.get_status()
#             iterations = cls.get_iterations()
#             print("Training after resubmit: status={} iterations={}".format(status, iterations))
#             sleep(2)
#         print('finish resubmit job')
#         bucket_test.remove()
