from __future__ import division, print_function, absolute_import
import os
import re
import nbformat
from jupyter_client import manager

try:
    from Queue import Empty
except ImportError:
    # python 3, so that file at least executable
    from queue import Empty

# Ignore these files
IGNORE = re.compile(r'.*ipykee.*')
# Also ignore ipython checkpoints
IGNORE_FOLDERS = re.compile(r'.*\.ipynb_checkpoints.*')

__author__ = 'Alex Rogozhnikov'


def check_single_notebook(notebook_filename, timeout=500):
    """
    Checks single notebook being given its full name.
    (executes cells one-by-one checking there are no exceptions, nothing more is guaranteed)
    """
    with open(notebook_filename) as notebook_file:
        notebook_content = nbformat.reads(notebook_file.read(), as_version=nbformat.current_nbformat)
        os.chdir(os.path.dirname(notebook_filename))
        _, client = manager.start_new_kernel()

        for cell in notebook_content.cells:
            if cell.cell_type == 'code':
                message_id = client.execute(cell.source)
                try:
                    message = client.get_shell_msg(message_id, timeout=timeout)
                except Empty:
                    raise RuntimeError("Cell timed out: \n {}".format(cell.source))

                if message['content']['status'] != 'ok':
                    traceback = message['content']['traceback']
                    description = "Cell failed: '{}'\n\n Traceback:\n{}".format(cell.source, '\n'.join(traceback))
                    raise RuntimeError(description)

        client.stop_channels()
