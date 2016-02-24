#!/bin/bash

# script that runs inside REP docker container to install modules that
# are found in specified folder (/etc_external/modules by default)
# Usage: $0 [modules_folder]


MODULES_DIR="/etc_external/modules"
[ -n "$1" ] && MODULES_DIR=$1
echo "Installing modules from $MODULES_DIR"
[ -d $MODULES_DIR ] || (echo "no $MODULES_DIR directory found" && exit)
for x in $MODULES_DIR/*; do
  if [ $x == "$MODULES_DIR/requirements.txt" ] ; then
    echo "processing $x"
    pip install -r $x
  else
    [ -d $x ] && pip install $x
  fi
done
