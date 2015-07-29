#!/bin/bash
halt() { echo $*
exit 1 
}
SYSTEM=`uname -s`

[ -z "$1" ] && halt "Usage: $0 DIRECTORY [NOTEBOOKDIR]"
REPDIR=$1 ; shift
[ -n "$1" ] && NOTEBOOKDIR=$1 && shift

which docker-compose > /dev/null
if [ $? -eq 1 ] ; then
	echo "Install docker-compose (http://docs.docker.com/compose/install/)"
	if [ $SYSTEM == "Darwin" ] ; then
	    echo "brew install docker-compose"
	else
	    echo "curl -L https://github.com/docker/compose/releases/download/1.3.2/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"
	    echo "chmod +x /usr/local/bin/docker-compose"
	fi
	exit 1
fi
HERE=`cd "$(dirname $0)" && pwd -P`
[ -d $REPDIR ] && halt "ERR: directory $REPDIR already exists"
mkdir -p $REPDIR || halt "Error creating $REPDIR"

cp -r $HERE/etc $REPDIR
find $HERE \( -name "*.sh" -o -name "*.yml" \) -a -not -name init.sh | xargs -I % cp % $REPDIR
[ -f $REPDIR/run.sh ] || halt "Error copying run.sh to $REPDIR"
for f in ipykee log modules; do
  mkdir -p $REPDIR/$f
done
if [ -n "$NOTEBOOKDIR" ] ; then
  ln -s $NOTEBOOKDIR $REPDIR/notebooks
else
  mkdir -p $REPDIR/notebooks
fi

echo "REPDIR is ready at '$REPDIR'"
echo "$REPDIR/run.sh"
