from __future__ import print_function, division, absolute_import

import os
import shutil
import requests

JSON_HEADER = {'Content-type': 'application/json'}

__author__ = 'Alexander Baranov'


class ServerError(Exception):
    pass


def check_result(result):
    result.raise_for_status()
    json = result.json()

    if json['success']:
        return json['data']
    else:
        raise ServerError(json['exception'])


def mn_post(*args, **kwargs):
    return check_result(requests.post(*args, **kwargs))


def mn_get(*args, **kwargs):
    return check_result(requests.get(*args, **kwargs))


def mn_delete(*args, **kwargs):
    return check_result(requests.delete(*args, **kwargs))


def mn_put(*args, **kwargs):
    return check_result(requests.put(*args, **kwargs))


class MatrixNetClient(object):
    def __init__(self, api_url, token):
        self.api_url = api_url
        self.bucket_kwargs = {'headers': {"X-Yacern-Token": token}}
        self.cls_kwargs = {'headers': {"X-Yacern-Token": token, 'Content-type': 'application/json'}}
        self.auth_token = token

    def bucket(self, **kwargs):
        return Bucket(self.api_url, requests_kwargs=self.bucket_kwargs, **kwargs)

    def classifier(self, **kwargs):
        return Estimator(api_url=self.api_url, classifier_type='mn', requests_kwargs=self.cls_kwargs, **kwargs)


class Bucket(object):
    """
    Bucket is a proxy for a dataset placed on the server.
    """

    def __init__(self, api_url, bucket_id=None, requests_kwargs=None):
        if requests_kwargs is None:
            requests_kwargs = {}
        self.api_url = api_url
        self.all_buckets_url = os.path.join(self.api_url, "buckets")
        self.requests_kwargs = requests_kwargs

        if bucket_id:
            self.bucket_id = bucket_id
            self.bucket_url = os.path.join(self.all_buckets_url, self.bucket_id)

            # Check if exists, create if does not.
            exists_resp = requests.get(self.bucket_url, **self.requests_kwargs)
            if exists_resp.status_code == 404:
                _ = mn_put(
                    self.all_buckets_url,
                    data={"bucket_id": self.bucket_id},
                    **self.requests_kwargs
                )
            else:
                exists_resp.raise_for_status()

        else:
            response = mn_put(self.all_buckets_url, **self.requests_kwargs)
            self.bucket_id = response['bucket_id']
            self.bucket_url = os.path.join(self.all_buckets_url, self.bucket_id)

    def ls(self):
        return mn_get(self.bucket_url, **self.requests_kwargs)

    def remove(self):
        return mn_delete(self.bucket_url, **self.requests_kwargs)

    def upload(self, local_filepath):
        files = {'file': open(local_filepath, 'rb')}

        result = mn_put(
            self.bucket_url,
            files=files,
            **self.requests_kwargs
        )

        return result['uploaded'] == 'ok'


class Estimator(object):
    def __init__(
            self,
            api_url,
            classifier_type, parameters, description, bucket_id,
            requests_kwargs=None
    ):
        """
        :param api_url: URL of server API
        :param classifier_type: string, for instance, 'mn'
        :param parameters: parameters of a classifier
        :param description: description of model
        :param bucket_id: associated bucked_id
        :param requests_kwargs: kwargs passed to request
        """
        if requests_kwargs is None:
            requests_kwargs = {'headers': JSON_HEADER}

        self.api_url = api_url
        self.all_cl_url = os.path.join(self.api_url, "classifiers")
        self.requests_kwargs = requests_kwargs
        self.status = None
        self._iterations = None
        self._debug = None

        self.classifier_type = classifier_type
        self.parameters = parameters
        self.description = description
        self.bucket_id = bucket_id

    def _update_with_dict(self, data):
        self.classifier_id = data['classifier_id']
        self.bucket_id = data['bucket_id']
        self.description = data['description']
        self.parameters = data['parameters']
        self.classifier_type = data['type']

    def _update_iteration_and_debug(self):
        response = mn_get(self._get_classifier_url_for('iterations'), **self.requests_kwargs)
        self._iterations = response.get('iterations')
        self._debug = response.get('debug')

    def _get_classifier_url(self):
        return os.path.join(self.all_cl_url, self.classifier_id)

    def _get_classifier_url_for(self, action):
        return os.path.join(self._get_classifier_url(), action)

    def load_from_api(self):
        data = mn_get(self._get_classifier_url(), **self.requests_kwargs)
        self._update_with_dict(data)

    def upload(self):
        payload = {
            'description': self.description,
            'type': self.classifier_type,
            'parameters': self.parameters,
            'bucket_id': self.bucket_id
        }

        data = mn_put(
            self.all_cl_url,
            json=payload,
            **self.requests_kwargs
        )
        self._update_with_dict(data)

        return True

    def get_status(self):
        self.status = mn_get(self._get_classifier_url_for('status'), **self.requests_kwargs)['status']
        return self.status

    def resubmit(self):
        self._iterations = None
        return mn_post(self._get_classifier_url_for('resubmit'), **self.requests_kwargs)['resubmit']

    def get_iterations(self):
        self._update_iteration_and_debug()
        return self._iterations

    def get_debug(self):
        self._update_iteration_and_debug()
        return self._debug

    def save_formula(self, path):
        response = requests.get(self._get_classifier_url_for('formula'), stream=True, **self.requests_kwargs)
        if not response.ok:
            raise ServerError('Error during formula downloading, {}'.format(response))

        with open(path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

    def save_stats(self, path):
        response = requests.get(self._get_classifier_url_for('stats'), stream=True, **self.requests_kwargs)
        if not response.ok:
            raise ServerError('Error during feature importances downloading, {}'.format(response))

        with open(path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
