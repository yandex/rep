#!/bin/bash

MYDIR=`cd "$(dirname $0)" && pwd -P`
source $MYDIR/_functions.sh

[ -z "$1" ] && halt "Usage: $0 REPDIR"
REPDIR=$1 ; shift
[ -d $REPDIR ] && halt "ERR: directory $REPDIR already exists"

mkdir -p $REPDIR || halt "Error creating directory $REPDIR"
cd $REPDIR

if [ "$1" == "--ln" ] ; then
  find $MYDIR -name "*.sh" -a -not -name "init.sh" | xargs -I % ln -s % .
else
  find $MYDIR -name "*.sh" -a -not -name "init.sh" | xargs -I % cp % .
fi
[ -f $REPDIR/run.sh ] || halt "Erorr creating 'run.sh' at $REPDIR"

for f in ipykee log modules notebooks ; do
  mkdir $f
done

echo "REPDIR is ready at '$REPDIR'. To start REP:"
echo "$REPDIR/run.sh"
