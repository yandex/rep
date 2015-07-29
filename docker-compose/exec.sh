#!/bin/bash

docker_cid() {
  image=$1
  [ -z "$image" ] && image="rep"
  SCRIPT_NAME=$0
  [ -z "SCRIPT_NAME" ] && SCRIPT_NAME=$BASH_SOURCE
  here=`cd $(dirname $SCRIPT_NAME) && pwd -P`
  prefix=`basename $here`
  echo "${prefix}_${image}_1"
}

CMD=bash
[ -n "$1" ] && CMD=$*

CID=`docker_cid rep`
docker exec -ti $CID $CMD
