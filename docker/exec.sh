#!/bin/bash

REPDIR=`cd "$(dirname $0)" && pwd -P`
source $REPDIR/_functions.sh
is_instance_dir $REPDIR || halt "ERROR: you should run $0 from REPDIR"
CID=`docker_cid rep`
docker exec -ti $CID $*
