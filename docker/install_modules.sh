#!/bin/bash

halt() { echo $*
exit 1
}


DIR=`cd "$(dirname 0)" && pwd -P`
PREFIX="/python"
cd $DIR
[ -d modules ] || halt "no modules directory found"
for x in modules/*; do 
  [ ! -d $x ] && continue
  echo $DIR/exec.sh "pip install ${PREFIX}_$x" 
  $DIR/exec.sh "pip install ${PREFIX}_$x" 
done
