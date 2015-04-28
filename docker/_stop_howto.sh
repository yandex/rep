#!/bin/bash
halt() { echo $*
exit 1
}

cid="REP_howto"
docker stop $cid
docker rm -v $cid
echo "Done"
