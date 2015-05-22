#!/bin/bash

NOTEBOOK_DIR='/notebooks'
[ -d '/logdir' ] && LOG_DIR='/logdir'

OPTIONS="--port=8080 --no-browser"
[ -n "$*" ] && OPTIONS=$*

ipython notebook $OPTIONS --ip=* --notebook-dir=$NOTEBOOK_DIR &>> $LOG_DIR/notebook.log
