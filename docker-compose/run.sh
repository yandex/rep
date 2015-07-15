#!/bin/bash

halt() {
echo $*
exit 1
}


HERE=`dirname $0`
CIDFILE=$HERE/rep.cid
[ ! -d "$HERE/notebooks" ] && halt "Are you running from the instance directory? (run \`./init.sh INSTANCE_DIR\`)"
[ -f $CIDFILE ] && rm $CIDFILE
cd $HERE
docker-compose up -d && echo "$(basename `pwd -L`)_rep_1" > $CIDFILE
