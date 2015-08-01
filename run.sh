#!/bin/bash

[ -f "/etc_docker/bash.bashrc" ] && source /etc_docker/bash.bashrc

[[ -L /notebooks/rep_howto ]] || ln -s /REP_howto /notebooks/rep_howto

NOTEBOOK_DIR='/notebooks'
[ -d '/logdir' ] && LOG_DIR='/logdir'

OPTIONS="--port=8080 --no-browser"
[ -n "$*" ] && OPTIONS=$*

ipython notebook $OPTIONS --ip=* --notebook-dir=$NOTEBOOK_DIR &>> $LOG_DIR/notebook.log
