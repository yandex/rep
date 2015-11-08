import os
import re
from rep.test.test_notebooks import check_single_notebook

import six

if six.PY3:
    # Notebooks are written in python2.
    import nose
    raise nose.SkipTest


# Ignore these files
IGNORE = re.compile(r'.*ipykee.*')
# Also ignore ipython checkpoints
IGNORE_FOLDERS = re.compile(r'.*\.ipynb_checkpoints.*')


def test_notebooks_in_folder(folder='../howto/'):
    dirname = os.path.dirname(os.path.realpath(__file__))
    howto_path = os.path.join(dirname, folder)
    for folder, _, files in os.walk(howto_path):
        if not IGNORE_FOLDERS.match(folder):
            for file_ in files:
                if file_.endswith(r".ipynb") and not IGNORE.match(file_):
                    print("Testing %s" % file_)
                    # False means not check
                    yield check_single_notebook, os.path.join(folder, file_)
