#!/bin/bash

halt() {
echo $*
exit 1
}


HERE=`dirname $0`
CIDFILE=$HERE/rep.cid
[ ! -d $HERE/notebooks ] && halt "Are you running from the instance directory?"
cd $HERE
docker-compose stop
[ -f $CIDFILE ] && rm -f $CIDFILE

