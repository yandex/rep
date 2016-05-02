# Makefile for building & starting rep-containers
# arguments can be supplied by -e definitions:
#
#  IMAGE -- name of image to use
#  N -- number of slave nodes 
#  HEADIP -- IP of head of the cluster
#

HEADIP ?= 10.16.23.10
DOCKER_NODE=docker -H tcp://0.0.0.0:2375
# NODE_OPTS=-H tcp://0.0.0.0:2375
PSSH=parallel-ssh
CLUSTER=cluster.txt
CLUSTER_SWARM=_cluster_swarm.txt
KERNELS_PER_NODE = 2
N_NODES ?= $(shell cat ${CLUSTER} | wc -l) # number of nodes in the cluster
N_KERNELS = $(shell echo $$(( ${N_NODES} * ${KERNELS_PER_NODE} )))
SWARM_OPT=-H tcp://localhost:31338
DOCKER_PACKAGE=docker-engine=1.10.3-0~wheezy
SWARM_IMAGE=swarm:1.1.3
IMAGE ?= yandex/rep-ipp:0.6.5
IMAGE_BUILD=yandex/rep-ipp:0.6.5
HERE:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/bash
MASTER=ipp_master
SHARED_HOME=/home/ipp
LOCAL_IMAGE_BUILD=${HEADIP}:5000/${IMAGE_BUILD}
SHARED_VOLUME=/soft
PASSWORD=kraCluster
HOSTNAME=kra.com
RUN_OPTIONS=-v ${SHARED_VOLUME}:/notebooks/${SHARED_VOLUME} -v ${SHARED_HOME}/security:/root/.ipython/profile_default/security -e PASSWORD=${PASSWORD} -e GENERATE_SSL_HOSTNAME=${HOSTNAME}


help:
	@echo Usage: make [-e VARIABLE=VALUE] targets
	@echo "variables:"
	@grep -h "#\s\+\w\+ -- " $(MAKEFILE_LIST) |sed "s/#\s//"
	@echo
	@echo targets and corresponding dependencies:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' -e 's/^/   /' | sed -e 's/##//'

${CLUSTER_SWARM}: ${CLUSTER}
	for i in `cat cluster.txt`; do \
		grep $$i /etc/hosts| awk '{print $$1":2375"}' ; done > ${CLUSTER_SWARM}
	# cat ${CLUSTER} | sed 's/$$/:2375/' > ${CLUSTER_SWARM}

lupdate-debian7-kernel: ## update kernel http://unix.stackexchange.com/questions/115898/how-do-i-upgrade-the-debian-wheezy-kernel-offline
	echo "deb http://ftp.pl.debian.org/debian wheezy-backports main" | tee -a /etc/apt/sources.list
	apt-get update
	aptitude -t wheezy-backports install linux-image-amd64
	reboot

pupdate-debian7-kernel: ${CLUSTER}
	${PSSH} -h ${CLUSTER} --timeout=0 -i " \
		echo 'deb http://ftp.pl.debian.org/debian wheezy-backports main' | tee -a /etc/apt/sources.list ; \
		apt-get update ; \
		aptitude -y -q -t wheezy-backports install linux-image-amd64"

puname: ${CLUSTER}
	${PSSH} -h ${CLUSTER} --timeout=0 -i "uname -a"

preboot: ${CLUSTER}
	${PSSH} -h ${CLUSTER} --timeout=0 -i "reboot"

pdocker-repo-debian7: ${CLUSTER} ## install docker Debian7 (https://docs.docker.com/engine/installation/linux/debian/#debian-wheezy-stable-7-x-64-bit)
	${PSSH} -h ${CLUSTER} --timeout=0 " \
		apt-get purge docker.io ; \
		apt-get -y -q install apt-transport-https ca-certificates ; \
		apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D ; \
		echo 'deb https://apt.dockerproject.org/repo debian-wheezy main' | sudo tee /etc/apt/sources.list.d/docker.list ; \
		apt-get update ; \
		apt-cache policy docker-engine && \
		apt-get install -y ${DOCKER_PACKAGE} "

pdocker-options: ${CLUSTER}
	${PSSH} -h ${CLUSTER} --timeout=0 " \
		echo 'DOCKER_OPTS=\"-H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock --insecure-registry ${HEADIP}:5000\"' | tee -a /etc/default/docker ; \
		service docker restart "


pdocker-repo-ubuntu:
	${PSSH} -h ${CLUSTER} -i 'echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list; cat /etc/apt/sources.list.d/docker.list'
	${PSSH} -h ${CLUSTER} 'sudo apt-get update'

pdocker-install: ${CLUSTER}
	${PSSH} -h ${CLUSTER} -i 'sudo apt-get install -y --force-yes docker-engine=1.9.1-0~trusty'

docker-versions: ${CLUSTER}
	${PSSH} -h ${CLUSTER} -i "${DOCKER_NODE} version"

pping: ## ping cluster
	${PSSH} -h ${CLUSTER} -i pwd

pclean-id: ${CLUSTER}
	${PSSH} -h ${CLUSTER} "rm /etc/docker/key.json && service docker restart"

puptime: ## uptime cluster
	${PSSH} -h ${CLUSTER} -i uptime

### SWARM 

_check-swarm-stopped:
	if [[ `docker ps | grep swarm` != "" ]] ; then echo "swarm master is already running" ; exit 1; fi

start-swarm: _check-swarm-stopped ${CLUSTER_SWARM} ## start swarm master
	docker run -v ${HERE}:/cfg -d -p 0.0.0.0:31338:2375 --restart=always --name=swarm_master ${SWARM_IMAGE} manage -H 0.0.0.0:2375 --strategy spread file:///cfg/${CLUSTER_SWARM}

stop-swarm: ## stop swarm master
	docker rm -f swarm_master

restart-swarm: stop-swarm start-swarm ## restart swarm master

swarm-info: ## check swarm
	docker ${SWARM_OPT} info

#### CLUSTER

start-master-interactive:
	docker run -p 32000-32100:32000-32100 --name ${MASTER} -ti --rm ${RUN_OPTIONS} ${LOCAL_IMAGE_BUILD} bash

start-master: ## start master
	docker run -p 32000-32100:32000-32100 -p 8888:8888 -e IPPHUB_IP=${HEADIP} --name ${MASTER} -d ${RUN_OPTIONS} ${LOCAL_IMAGE_BUILD}
	( echo -e "GET /\n\n" | nc localhost 8888 ) || exit 1

start-slaves: ## start slaves
	sleep 2
	for i in `seq ${N_KERNELS}` ; do \
		echo "start slave $$i" ; \
		docker ${SWARM_OPT} run -d ${RUN_OPTIONS} ${LOCAL_IMAGE_BUILD} bash --login -c ipengine || exit 1 ; \
	done
	echo "Running `docker ${SWARM_OPT} ps -q |wc -l` slave(s)"

is-master-running:
	docker ps -a | grep "${MASTER}$$" > /dev/null
	( echo -e "GET /\n\n" | nc localhost 8888 ) || exit 1

test-cluster: is-master-running list-slaves ## test parallel script 
	docker cp tests ${MASTER}:/root
	docker exec ${MASTER} bash --login -c 'for p in tests/* ; do echo $$p ; python $$p || exit 1 ; done'

exec-master: is-master-running
	docker exec -ti ${MASTER} bash 

start-cluster: start-master start-slaves list-slaves test-cluster ## start cluster

restart-cluster: stop-cluster start-cluster ## restart cluster

list-slaves:  ## list slaves
	docker ${SWARM_OPT} ps --no-trunc

stop-slaves:
	docker ${SWARM_OPT} ps -q | xargs --no-run-if-empty docker ${SWARM_OPT} rm -f

stop-master: is-master-running
	docker stop ${MASTER}
	docker rm ${MASTER}

restart-slaves: stop-slaves start-slaves  ## restart slaves
	echo Slaves restarted

stop-cluster: stop-slaves stop-master
	echo Cluster stopped

#### REGISTRY

start-registry:
	test -d ${HERE}/registry || mkdir -p ${HERE}/registry
	docker run -d -p 5000:5000 --restart=always --name registry -v ${HERE}/registry:/var/lib/registry registry:2 
	echo http://${HEADIP}:5000

stop-registry:
	docker stop registry
	docker rm -f registry

list-registry:
	# https://docs.docker.com/registry/spec/api/
	curl http://10.16.23.10:5000/v2/_catalog

build-image:
	docker build -t ${IMAGE_BUILD} .
	docker tag ${IMAGE_BUILD} ${LOCAL_IMAGE_BUILD}

push-image:
	docker push ${LOCAL_IMAGE_BUILD}

pull-image-swarm:
	docker ${SWARM_OPT} pull ${LOCAL_IMAGE_BUILD}


##### Obsolete?

ps-user-containers: ${CLUSTER} ## list container running on the cluster
	${PSSH} -h ${CLUSTER} -i '${DOCKER_NODE} ps '

images: ${CLUSTER} ## list images created at clusters
	${PSSH} -h ${CLUSTER} -i '${DOCKER_NODE} images'

rm-images: ${CLUSTER} ## remove all images
	${PSSH} -h ${CLUSTER} -i '${DOCKER_NODE} images -q | xargs ${DOCKER_NODE} rmi'

rm-user-containers: ${CLUSTER} ## stop & remove user containers
	${PSSH} -h ${CLUSTER} --timeout=0 -i '${DOCKER_NODE} ps -a -q|xargs --no-run-if-empty ${DOCKER_NODE} rm'

swarm-ps: ## list containers running in swarm
	docker ${SWARM_OPT} ps

swarm-psa: ## list all containers in swarm
	docker ${SWARM_OPT} ps -a

start-monitor: ## start monitoring (bosun)
	docker -p 0.0.0.0:4242:4242 -p 0.0.0.0:8070:8070 stackexchange/bosun

pdf: ${CLUSTER} ## check disk free space on cluster nodes
	${PSSH} -h ${CLUSTER} -i df -h /

mdu: ## mfs du
	du -m --max-depth 1 /opt/mfs
