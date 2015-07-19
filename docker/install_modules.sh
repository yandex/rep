#!/bin/bash

REPDIR=`cd "$(dirname $0)" && pwd -P`
source $REPDIR/_functions.sh

PREFIX="/python"
cd $REPDIR
[ -d modules ] || halt "no modules directory found"
for x in modules/*; do 
  if [ $x == "modules/requirements.txt" ] ; then
    echo "processing $x" 
    $REPDIR/exec.sh "pip install -r ${PREFIX}_$x"
    continue
  fi
  [ ! -d $x ] && continue
  echo $REPDIR/exec.sh "pip install ${PREFIX}_$x" 
  $REPDIR/exec.sh "pip install ${PREFIX}_$x" 
done
