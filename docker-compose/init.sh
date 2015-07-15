#!/bin/bash
halt() { echo $*
exit 1 
}

which docker-compose > /dev/null
if [ $? -eq 1 ] ; then
	echo "Install docker-compose (http://docs.docker.com/compose/install/)"
	echo "curl -L https://github.com/docker/compose/releases/download/1.3.2/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"
	echo "chmod +x /usr/local/bin/docker-compose"
	exit 1
fi
CP_CMD=cp
MYDIR=`cd "$(dirname $0)" && pwd -P`
[ -z "$1" ] && halt "Usage: $0 DIRECTORY"
DIR=$1 ; shift
[ "$1" == "--ln" ] && CP_CMD="ln -s" && shift
[ -d $DIR ] && halt "ERR: directory $DIR already exists"
mkdir -p $DIR

$CP_CMD -r $MYDIR/etc $DIR
$CP_CMD $MYDIR/run.sh $DIR
$CP_CMD $MYDIR/exec.sh $DIR 
$CP_CMD $MYDIR/docker-compose.yml $DIR
for f in ipykee log modules notebooks ; do
  mkdir -p $DIR/$f
done

echo "instance dir is ready at '$DIR'"
echo "cd $DIR"
echo "./run.sh"
