#!/bin/sh

NOTEBOOK_DIR='/notebooks'
[ -d '/logdir' ] && LOG_DIR='/logdir'

ipython notebook --ip=* --port=8080 --pylab inline --no-browser --notebook-dir=$NOTEBOOK_DIR &>> $LOG_DIR/notebook.log
