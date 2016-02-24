#!/bin/bash

TESTS_MASK=""
# TESTS_MASK="z_test_notebook"

# finding directory
HERE=$(cd `dirname $0`; pwd)
cd $HERE

export NOSETESTS_OPTIONS="-vd --nocapture -x"

nosetests $NOSETESTS_OPTIONS $TESTS_MASK
# standard way of getting results status
RESULT_STATUS=$?

# listing all the files recursively in howto folder
ls -alR ../howto

exit $RESULT_STATUS
