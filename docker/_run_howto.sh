#!/bin/bash

halt() { echo $*
exit 1
}

IMAGE=anaderi/rep-howto:latest
docker pull $IMAGE || halt "unable to pull $IMAGE"
docker run -d --name REP_howto $IMAGE
