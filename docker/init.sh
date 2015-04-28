#!/bin/bash
halt() { echo $*
exit 1 
}

CP_CMD=cp
MYDIR=`cd "$(dirname 0)" && pwd -P`
[ -z "$1" ] && halt "Usage: $0 DIRECTORY"
DIR=$1 ; shift
[ "$1" == "--ln" ] && CP_CMD="ln -s" && shift
[ -d $DIR ] && halt "ERR: directory $DIR already exists"
mkdir -p $DIR

$CP_CMD $MYDIR/*.sh $DIR
for f in ipykee log modules notebooks ; do
  mkdir $DIR/$f
done

echo "REPrc dir is ready at '$DIR'"
echo "cd $DIR"
echo "./run.sh"
