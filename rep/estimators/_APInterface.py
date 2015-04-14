from __future__ import division, print_function, absolute_import
import os
import requests

__author__ = 'Tatiana Likhomanenko'

__version__ = 0.4

VERBOSE = False
CHUNK_SIZE = 128


class API_interface:
    def __init__(self, connection):
        """
        :param str connection: name of string in config file which is situated in .: connection: url, username, password
        """
        self.connection = connection

    def _auth(self):
        try:
            import ConfigParser
        except ImportError:
            # python 3
            import configparser as ConfigParser

        config = ConfigParser.RawConfigParser()
        config.read('config')
        configParameters = config.defaults()
        url, user, password = configParameters[self.connection].split(',')
        url, user, password = url.strip(), user.strip(), password.strip()
        user = None if user == 'None' else user
        password = None if password == 'None' else password

        from requests.auth import HTTPDigestAuth
        return url, HTTPDigestAuth(user, password)

    def _exec_post(self, url, data, files=None):
        """
        Post request

        :param str url: url for post request
        :param dict(str: str) data: parameters
        :param dict(str: file) files: files

        :return: parsed json result
        """
        url_base, auth = self._auth()
        request_post = requests.post(os.path.join(url_base, url), data=data, files=files, auth=auth)
        try:
            json_result = request_post.json()
        except ValueError:
            raise Exception("error parsing JSON:" + request_post.text)
        return json_result

    def _exec_get(self, url, parameters, outfile=None):
        """

        :param str url: url for post request
        :param dict(str: str) parameters: parameters
        :param str outfile: file for request output

        :return: filename or parsed json
        """
        url_base, auth = self._auth()
        request_get = requests.get(os.path.join(url_base, url), params=parameters, stream=outfile is not None,
                                   auth=auth)
        if outfile is not None:
            assert request_get.ok, 'Something wrong in request'
            assert 'content-disposition' in request_get.headers and "attachment; filename" in request_get.headers[
                'content-disposition'], \
                'Error during download file, {}, {}'.format(request_get.headers, request_get.text)
            with open(outfile, "wb") as fp:
                for chunk in request_get.iter_content(CHUNK_SIZE):
                    fp.write(chunk)
            assert os.path.exists(outfile), "oops, file still not there"
            result = outfile
        else:
            try:
                result = request_get.json()
            except ValueError:
                raise Exception("error parsing JSON:" + request_get.text)
        return result
