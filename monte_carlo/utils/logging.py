import os
import sys
import datetime
from typing import Optional, TextIO

from ..resources import filepaths
import warnings


out_file: Optional[TextIO]


def create_new_log_dir():
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    filepath = os.path.join(os.path.dirname(__file__), filepaths.text_log_path + time_string + '/')
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath


def start_logging(logging_category='general'):
    assert logging_category is not None
    print('Logging started. '+str(logging_category))
    filename = logging_category+'.txt'

    global log_dir
    global out_file
    global orig_stdout
    global category
    category = logging_category

    if out_file is not None or orig_stdout is not None:
        raise PermissionError('Starting multiple logging sessions is not allowed.')

    if log_dir is None:
        log_dir = create_new_log_dir()

    orig_stdout = sys.stdout

    out_file = open(log_dir+filename, 'a')
    sys.stdout = out_file


def stop_logging():
    global out_file
    global orig_stdout
    global category
    if out_file is None or orig_stdout is None:
        warnings.warn('Cannot stop logging before starting.')
        return
    sys.stdout = orig_stdout
    print('Logging stopped. '+str(category))
    orig_stdout = None
    out_file.close()
    out_file = None


def switch_logging_category(logging_category):
    if out_file is not None and category is not None:
        global previous_category
        previous_category = category
        stop_logging()
    start_logging(logging_category)


def switch_to_previous_category():
    if previous_category is not None:
        switch_logging_category(previous_category)
    else:
        stop_logging()
        print('No previous category. Stopping logging session. {}'.format(switch_logging_category.__name__))


out_file = None
orig_stdout = None
log_dir = None
category = None
previous_category = None
