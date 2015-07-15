#!/bin/bash

HERE=`dirname $0`
[ -f $HERE/rep.cid ] && CID=`cat $HERE/rep.cid`
[[ -z "$1" && -z "$CID" ]] && echo "Usage: $0 CONTAINER_ID" && exit 1
[ -n "$1" ] && CID=$1
docker exec -ti $CID bash
