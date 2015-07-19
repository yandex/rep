#!/bin/bash

usage() {
  echo "Usage: $0 [--howto] [port#]"
  echo
  echo "Example: $0 --howto 8000"
  exit
}

IMAGE="anaderi/rep:latest"
REPDIR=`cd "$(dirname $0)" && pwd -P`
source $REPDIR/_functions.sh
protocol=http

port=8080
nbdiff_port=8086

[ "$1" = "-h" ] && usage
[ -n "$1" ] && port=$1 && shift

CID=`docker_cid rep`
is_container_running $CID && halt "Container $CID is already running"

REPDIR_MAP="-v $REPDIR/notebooks:/notebooks -v $REPDIR/log:/logdir -v $REPDIR/ipykee:/ipykee -v $REPDIR/modules:/python_modules"
[ -d "/afs" ] && REPDIR_MAP+=" -v /afs:/afs"

PORT_MAP="-p $port:8080 -p $nbdiff_port:5000"

docker run --net=host --name $CID -d $PORT_MAP $REPDIR_MAP $IMAGE || halt "Error starting docker"
echo "Installing modules"
$REPDIR/install_modules.sh || echo "Warning: not all modules installed correctly"
echo "Container $CID has created. URL: $protocol://`get_rep_host`:$port"
