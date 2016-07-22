import os
from rep.test.test_notebooks import check_single_notebook

import six

if six.PY3:
    # Notebooks are written in python2, so not testing it under python 3
    import nose
    raise nose.SkipTest


def test_notebooks_in_folder(folder='../howto/'):
    dirname = os.path.dirname(os.path.realpath(__file__))
    howto_path = os.path.join(dirname, folder)
    for file_name in os.listdir(howto_path):
        if file_name.endswith(r".ipynb"):
            print("Testing %s" % file_name)
            yield check_single_notebook, os.path.join(howto_path, file_name)
