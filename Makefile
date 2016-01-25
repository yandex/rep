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


include .rep_version  # define REP_BASE_IMAGE, and REP_IMAGE

.PHONY: run rep-image rep-base-image inspect

version:
	@echo $(REP_IMAGE), $(REP_BASE_IMAGE)

rep-base-image:
	source .version && docker build -t $(REP_BASE_IMAGE) -f ci/Dockerfile.base ci

rep-base-image3:
	TRAVIS_PYTHON_VERSION=3.4 docker build -t $(REP_BASE_IMAGE) -f ci/Dockerfile.base ci

rep-image:
	docker build -t $(REP_IMAGE) -f ci/Dockerfile.rep .

local-dirs:
	[ -d $(NOTEBOOKS) ] || mkdir -p $(NOTEBOOKS)
	[ -d $(ETC) ] || mkdir -p $(ETC)

run: local-dirs
	docker run -ti --rm $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE) $(RUN_ARGS) 

run-daemon: local-dirs
	docker run -d $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE) $(RUN_ARGS) 

restart:
	docker restart $(CONTAINER_NAME)

exec:
	docker exec -ti $(CONTAINER_NAME) $(EXEC_ARGS)

logs:
	docker logs $(CONTAINER_NAME)

stop:
	docker stop $(CONTAINER_NAME)

remove: stop
	docker rm $(CONTAINER_NAME)

inspect:
	docker inspect $(REP_IMAGE) 

push: rep-image
	@docker login -e="$(DOCKER_EMAIL)" -u="$(DOCKER_USERNAME)" -p="$(DOCKER_PASSWORD)"
	docker push $(REP_IMAGE)

push-base:
	docker push $(REP_BASE_IMAGE)

tag-latest: rep-image
	docker tag -f $(REP_IMAGE) yandex/rep:latest

push-latest: tag-latest push
	docker push yandex/rep:latest
