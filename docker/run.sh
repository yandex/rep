#!/bin/bash

halt() { 
  echo $* 
  exit 1 
}

usage() {
  echo "Usage: $0 [--howto] [port#]"
  echo
  echo "Example: $0 --howto 8000"
  exit
}

IMAGE="anaderi/rep:latest"
DIR=`cd "$(dirname 0)" && pwd -P`
CID_FILE=$DIR/docker.cid
port=8080
nbdiff_port=8086

[ "$1" = "-h" ] && usage
if [ "$1" = "--howto" ] ; then
  $DIR/_run_howto.sh || halt "unable to run REP_howto"
  shift
fi
[ -n "$1" ] && port=$1 && shift

[ -f $CID_FILE ] && halt "CID file exists ($CID_FILE, $(cat $CID_FILE)). Is REP already running?"

DIR_MAP="-v $DIR/notebooks:/notebooks -v $DIR/log:/logdir -v $DIR/ipykee:/ipykee -v $DIR/modules:/python_modules"
[ -d "/afs" ] && DIR_MAP+=" -v /afs:/afs"
docker ps -a|grep "REP_howto\s*\$" > /dev/null
[ $? -eq 0 ] && DIR_MAP+=" --volumes-from REP_howto"
PORT_MAP="-p $port:8080 -p $nbdiff_port:5000"

CID=$(docker run  --net=host -d $PORT_MAP $DIR_MAP $IMAGE)
[ -z "$CID" ] && halt "Error starting docker"
echo $CID > $CID_FILE
echo "Done. CID=$CID" | head -c 23
echo
