from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

import functools
import nose


def known_failure(test):
    """
    Decorator to mark known failures in tests
    """
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception:
            raise nose.SkipTest

    return inner


def retry_if_fails(test):
    """
    Decorator to mark tests which can frequently fail.
    """
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except AssertionError:
            test(*args, **kwargs)

    return inner