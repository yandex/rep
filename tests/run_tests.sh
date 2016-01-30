#!/bin/bash

RUN_TESTS=""
#RUN_TESTS="z_test_notebook"

HERE=$(cd `dirname $0`; pwd)
cd $HERE

#export TEST_NOTEBOOKS_REGEX=01
#export SKIP_NOTEBOOKS_REGEX=intro
export OPTIONS="-vd --nocapture -x"
#export OPTIONS+=" --collect-only"  # dry-run

nosetests $OPTIONS $RUN_TESTS | grep -v "downhill.base:232" 
RESULT=${PIPESTATUS[0]} 
ls -alR ../howto

exit $RESULT
