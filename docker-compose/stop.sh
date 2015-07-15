#!/bin/bash

HERE=`dirname $0`
CIDFILE=$HERE/rep.cid
cd $HERE
docker-compose stop
[ -f $CIDFILE ] && rm -f $CIDFILE

