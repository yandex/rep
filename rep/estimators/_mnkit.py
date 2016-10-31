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
        return Estimator(self.api_url, requests_kwargs=self.cls_kwargs, **kwargs)


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
            self, api_url,
            cl_id=None,
            cl_type="mn", parameters=None, description=None, bucket_id=None,
            requests_kwargs=None
    ):
        if requests_kwargs is None:
            requests_kwargs = {'headers': JSON_HEADER}
        self.api_url = api_url
        self.all_cl_url = os.path.join(self.api_url, "classifiers")
        self.requests_kwargs = requests_kwargs
        self.status = None
        self._iterations = None
        self._debug = None

        if cl_id:
            self.cl_id = cl_id
            self.cl_url = os.path.join(self.all_cl_url, self.cl_id)
            self.load_from_api()
        elif all((cl_type, parameters, description, bucket_id)):
            self.cl_type = cl_type
            self.parameters = parameters
            self.description = description
            self.bucket_id = bucket_id
        else:
            raise Exception("Neither cl_id nor estimator parameters are specified")

    def _update_with_dict(self, data):
        self.cl_id = data['classifier_id']
        self.cl_url = os.path.join(self.all_cl_url, self.cl_id)

        self.bucket_id = data['bucket_id']
        self.description = data['description']
        self.parameters = data['parameters']
        self.cl_type = data['type']

    def _update_iteration_and_debug(self):
        response = mn_get(os.path.join(self.cl_url, 'iterations'), **self.requests_kwargs)
        self._iterations = response.get('iterations')
        self._debug = response.get('debug')

    def load_from_api(self):
        self.cl_url = os.path.join(self.all_cl_url, self.cl_id)
        data = mn_get(self.cl_url, **self.requests_kwargs)
        self._update_with_dict(data)

    def upload(self):
        payload = {
            'description': self.description,
            'type': self.cl_type,
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
        self.status = mn_get(os.path.join(self.cl_url, 'status'), **self.requests_kwargs)['status']
        return self.status

    def resubmit(self):
        self._iterations = None
        return mn_post(os.path.join(self.cl_url, 'resubmit'), **self.requests_kwargs)['resubmit']

    def get_iterations(self):
        self._update_iteration_and_debug()
        return self._iterations

    def get_debug(self):
        self._update_iteration_and_debug()
        return self._debug

    def save_formula(self, path):
        response = requests.get(os.path.join(self.cl_url, 'formula'), stream=True, **self.requests_kwargs)
        if not response.ok():
            raise ServerError('Error during formula downloading, {}'.format(response))

        with open(path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

    def save_stats(self, path):
        response = requests.get(os.path.join(self.cl_url, 'stats'), stream=True, **self.requests_kwargs)
        if not response.ok():
            raise ServerError('Error during feature importances downloading, {}'.format(response))

        with open(path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
