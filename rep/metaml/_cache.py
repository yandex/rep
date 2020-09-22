from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

import os
import datetime

import base64
from six.moves import cPickle


def get_folder_size(folder):
    """
    Compute the total size of folder in cross-platform way
    """
    total_size = 0
    for dir_path, _, file_names in os.walk(folder):
        for filename in file_names:
            fp = os.path.join(dir_path, filename)
            total_size += os.path.getsize(fp)
    return total_size


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


class CacheHelper(object):
    def __init__(self, folder, expiration_in_seconds):
        self.folder = os.path.abspath(folder)
        self.expiration_in_seconds = expiration_in_seconds

    def initialize_cache(self):
        """ Creates a folder for cache """
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        # Delete too old files
        for filename in os.listdir(self.folder):
            file_path = os.path.join(self.folder, filename)
            modified_date = modification_date(file_path)
            if datetime.datetime.now() - modified_date > datetime.timedelta(seconds=self.expiration_in_seconds):
                os.remove(file_path)

    def clear_cache(self):
        """ Deletes the cache folder """
        if os.path.exists(self.folder):
            import shutil
            shutil.rmtree(self.folder)

    def _get_filename(self, key):
        file_name_string = base64.urlsafe_b64encode(key.encode('ascii')).decode('ascii') + '.pkl'
        return os.path.join(self.folder, file_name_string)

    def store_in_cache(self, key, control_hash, value):
        self.initialize_cache()
        file_name_string = self._get_filename(key)

        with open(file_name_string, mode='wb') as f:
            cPickle.dump([control_hash, value], f)

    def get_from_cache(self, key, control_hash):
        self.initialize_cache()
        filename = self._get_filename(key)
        if not os.path.isfile(filename):
            return False, None
        with open(filename, 'rb') as f:
            stored_control_hash, value = cPickle.load(f)
            if stored_control_hash == control_hash:
                return True, value
            else:
                return False, None
