#!/bin/bash

[ -f "/etc_docker/bash.bashrc" ] && source /etc_docker/bash.bashrc

NOTEBOOK_DIR='/notebooks'
[ -d '/logdir' ] && LOG_DIR='/logdir'

OPTIONS="--port=8080 --no-browser"
[ -n "$*" ] && OPTIONS=$*

ipython notebook $OPTIONS --ip=* --notebook-dir=$NOTEBOOK_DIR &>> $LOG_DIR/notebook.log
