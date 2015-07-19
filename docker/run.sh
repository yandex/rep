#!/bin/bash

usage() {
  echo "Usage: $0 [port#]"
  echo
  echo "Example: $0 8000"
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

is_instance_dir $REPDIR || halt "ERROR: you should run $0 from REPDIR"
CID=`docker_cid rep`
is_container_running $CID && halt "Container $CID is already running"
is_container_created $CID 
if [ $? -eq 1 ] ; then
    REPDIR_MAP="-v $REPDIR/notebooks:/notebooks -v $REPDIR/log:/logdir -v $REPDIR/ipykee:/ipykee -v $REPDIR/modules:/python_modules"
    [ -d "/afs" ] && REPDIR_MAP+=" -v /afs:/afs"

    PORT_MAP="-p $port:8080 -p $nbdiff_port:5000"

    docker run --net=host --name $CID -d $PORT_MAP $REPDIR_MAP $IMAGE || halt "Error starting docker image"
    $REPDIR/install_modules.sh || echo "Warning: not all modules installed correctly"
else
    echo "Restarting container $CID"
    docker restart $CID > /dev/null || halt "Error resuming container"
fi
echo "Container $CID is ready. URL: $protocol://`get_rep_host`:$port"
