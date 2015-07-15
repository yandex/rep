#!/bin/bash

HERE=`dirname $0`
CIDFILE=$HERE/rep.cid
[ -f $CIDFILE ] && rm $CIDFILE
docker-compose up -d && echo "$(basename `pwd -L`)_rep_1" > $CIDFILE
