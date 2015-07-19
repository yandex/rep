#!/bin/bash

halt() {
echo $*
exit 1
}


HERE=`dirname $0`
[ ! -d $HERE/notebooks ] && halt "Are you running from the instance directory?"
cd $HERE
docker-compose rm
