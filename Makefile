# Makefile for building & starting rep-containers
# arguments can be supplied by -e definitions: 
#
#    ENV -- filename with environment variables passed to docker container
#    NOTEBOOKS -- folder with notebooks that will be mounted into docker container
#    PORT -- port to listen for incoming connection
#    ETC -- local folder that will be moundted to /etc_external into docker container
#
#
ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif
ifeq (exec,$(firstword $(MAKECMDGOALS)))
  EXEC_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(EXEC_ARGS):;@:)
endif
ifeq   "$(EXEC_ARGS)" ""
  EXEC_ARGS := bash
endif
CONTAINER_NAME := $(shell basename $(CURDIR) | tr - _ )
NOTEBOOKS ?= $(shell pwd)/notebooks
ETC ?= $(shell pwd)/etc
VOLUMES := -v $(ETC):/etc_external -v $(NOTEBOOKS):/notebooks
PORT ?= 8888
DOCKER_ARGS := $(VOLUMES) -p $(PORT):8888
ifneq "$(ENV)" ""
  DOCKER_ARGS := $(DOCKER_ARGS) --env-file=$(ENV)
endif


include .rep_version  # read REP_BASE_IMAGE, and REP_IMAGE

help:
	@echo Makefile targets and corresponding dependencies:
	@echo
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: run rep-image rep-base-image inspect

version:
	@echo $(REP_IMAGE), $(REP_BASE_IMAGE)

rep-base-image:	## build REP base image
	source .version && docker build -t $(REP_BASE_IMAGE) -f ci/Dockerfile.base ci

rep-base-image3:
	TRAVIS_PYTHON_VERSION=3.4 docker build -t $(REP_BASE_IMAGE) -f ci/Dockerfile.base ci

rep-image:	## build REP image
	docker build -t $(REP_IMAGE) -f ci/Dockerfile.rep .

local-dirs:
	[ -d $(NOTEBOOKS) ] || mkdir -p $(NOTEBOOKS)
	[ -d $(ETC) ] || mkdir -p $(ETC)

run: local-dirs		## run REP interactively
	docker run -ti --rm $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE) $(RUN_ARGS) 

run-daemon: local-dirs	## run REP as a daemon
	docker run -d $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE) $(RUN_ARGS) 

restart:	## restart REP container
	docker restart $(CONTAINER_NAME)

exec:		## run command within REP container
	docker exec -ti $(CONTAINER_NAME) $(EXEC_ARGS)

logs:		## show container logs
	docker logs $(CONTAINER_NAME)

stop:		## stop REP container
	docker stop $(CONTAINER_NAME)

remove: stop	## remove REP container
	docker rm $(CONTAINER_NAME)

inspect:	## inspect REP image
	docker inspect $(REP_IMAGE) 

push: rep-image	## build REP image & push to docker hub
	@docker login -e="$(DOCKER_EMAIL)" -u="$(DOCKER_USERNAME)" -p="$(DOCKER_PASSWORD)"
	docker push $(REP_IMAGE)

push-base:	## push base image to docker hub
	docker push $(REP_BASE_IMAGE)

tag-latest: rep-image	## tag current REP image as latest
	docker tag -f $(REP_IMAGE) yandex/rep:latest

push-latest: tag-latest push	## tag current REP image as latest and push it to docker hub
	docker push yandex/rep:latest
