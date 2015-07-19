#!/bin/bash

# functions for running REP under docker
SYSTEM=`uname -s`

halt() {
  echo $*
  exit 1
}

docker_cid() {
  image=$1
  [ -z "$image" ] && image="rep"
  SCRIPT_NAME=$0
  [ -z "SCRIPT_NAME" ] && SCRIPT_NAME=$BASH_SOURCE
  here=`cd $(dirname $SCRIPT_NAME) && pwd -P`
  prefix=`basename $here`
  echo "${prefix}_${image}"
}

is_container_running() {
  cid=$1
  docker ps -a|grep $cid > /dev/null
  return $?
}

get_rep_host() {
  host=localhost
  if [ "$SYSTEM" == "Darwin" ] ; then
    host=`boot2docker ip`
  fi
  echo $host
}

