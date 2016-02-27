#!/bin/bash

# script that runs inside REP docker container to install modules that
# are found in specified folder (/etc_external/modules by default)
# Usage: $0 [modules_folder]


REP_MODULES_DIR="/etc_external/modules"
[ -n "$1" ] && REP_MODULES_DIR=$1
echo "Installing modules from $REP_MODULES_DIR"
[ -d $REP_MODULES_DIR ] || (echo "no $REP_MODULES_DIR directory found" && exit)
for x in $REP_MODULES_DIR/*; do
  if [ $x == "$REP_MODULES_DIR/requirements.txt" ] ; then
    echo "processing $x"
    pip install -r $x
  else
    [ -d $x ] && pip install $x
  fi
done
