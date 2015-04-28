#!/bin/bash
halt() { echo $*
exit 1
}

DIR=`cd "$(dirname 0)" && pwd -P`
CID_FILE=$DIR/docker.cid
[ ! -f $CID_FILE ] && halt "file with CID not found"
cid=`cat $CID_FILE`
[ -z "$cid" ] && halt "Invalid CID: '$cid'"
docker exec $cid $*
