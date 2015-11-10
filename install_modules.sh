#!/bin/bash

halt() {
	echo $*
	exit
}

MODULES_DIR="/etc_external/modules"
echo "Installing modules from $MODULES_DIR"
[ -d $MODULES_DIR ] || halt "no $MODULES_DIR directory found"
for x in $MODULES_DIR/*; do
  if [ $x == "$MODULES_DIR/requirements.txt" ] ; then
    echo "processing $x"
    pip install -r $x
    continue
  fi
  [ -d $x ] && pip install $x
done