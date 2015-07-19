#!/bin/bash

REPDIR=`cd "$(dirname $0)" && pwd -P`
source $REPDIR/_functions.sh
CID=`docker_cid rep`
docker exec -ti $CID $*
