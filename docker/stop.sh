#!/bin/bash
halt() { echo $*
exit 1
}

usage() {
  echo "Usage: $0 [--howto]"
  echo
  echo "Example: $0 --howto"
  exit
}

DIR=`cd "$(dirname 0)" && pwd -P`
CID_FILE=$DIR/docker.cid
[ ! -f $CID_FILE ] && halt "file with CID not found"
cid=`cat $CID_FILE`
[ -z "$cid" ] && halt "Invalid CID: '$cid'"
[ "$1" = "-h" ] && usage
docker stop $cid
docker rm -v $cid
rm $CID_FILE
if [ "$1" = "--howto" ] ; then
  $DIR/_stop_howto.sh || halt "unable to stop REP_howto"
  shift
fi
echo "Done"
