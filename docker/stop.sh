#!/bin/bash

usage() {
  echo "Usage: $0 [--howto]"
  echo
  echo "Example: $0 --howto"
  exit
}

REPDIR=`cd "$(dirname $0)" && pwd -P`
source $REPDIR/_functions.sh

CID=`docker_cid rep`
[ "$1" = "-h" ] && usage
docker stop $CID > /dev/null || halt "Error stopping container $CID"
docker rm -v $CID > /dev/null || halt "Erorr removing container $CID"
echo "Done"
