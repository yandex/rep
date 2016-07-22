# Makefile for building & starting REP containers
# arguments can be supplied by -e definitions: 
#
#	NOTEBOOKS -- folder to be mounted to container (default: ./notebooks)
#	PORT -- port to listen for incoming connection (default: 8888)
#

CONTAINER_NAME := rep
NOTEBOOKS ?= $(shell pwd)/notebooks
PORT ?= 8888
DOCKER_ARGS := --volume $(NOTEBOOKS):/notebooks -p $(PORT):8888

include .rep_version  # read REP_IMAGE

help:
	@echo Usage: make [-e VARIABLE=VALUE] targets
	@echo "variables:"
	@grep -h "#\s\+\w\+ -- " $(MAKEFILE_LIST) |sed "s/#\s//"
	@echo
	@echo targets:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' -e 's/^/   /' | sed -e 's/##//'

.PHONY: run rep-image2 rep-image3 run-daemon restart logs  \
	inspect exec help stop remove push push-base tag-latest push-latest

version:
	@echo $(REP_IMAGE)

rep-image2:	## build REP image with python 2
	docker build --build-arg REP_PYTHON_VERSION=2 -t $(REP_IMAGE) -f ci/Dockerfile.rep .

rep-image3:	## build REP image with python 3
	docker build --build-arg REP_PYTHON_VERSION=3 -t $(REP_IMAGE) -f ci/Dockerfile.rep .

local-dirs: # creates a local directory to be mounted to REP container
	[ -d $(NOTEBOOKS) ] || mkdir -p $(NOTEBOOKS)

run: local-dirs		## run REP interactively
	docker run --interactive --tty --rm $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE)

run-daemon: local-dirs	## run REP as a daemon
	docker run --detach $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE)

run-tests:  ## run tests inside a container, both notebooks and scripts
	# for some reason nosetests fails if directly mounted to tests folder
	mkdir -p ./notebooks/tests
	cp -r tests ./notebooks/tests
	find tests -name '*.pyc' -delete
	docker run  --interactive --tty --rm --volume $(shell pwd)/notebooks:/notebooks $(REP_IMAGE) \
		/bin/bash -l -c "cd /notebooks/tests && nosetests -v --detailed-errors --nocapture . "

restart:	## restart REP container
	docker restart $(CONTAINER_NAME)

exec:       ## run command within REP container
	docker exec -ti $(CONTAINER_NAME)

show-logs:  ## show container logs
	docker logs $(CONTAINER_NAME)

stop:       ## stop REP container
	docker stop $(CONTAINER_NAME)

remove: stop    ## remove REP container
	docker rm $(CONTAINER_NAME)

inspect:	# inspect REP image
	docker inspect $(REP_IMAGE)

push: rep-image2	# build REP image & push to docker hub
	@docker login -e="$(DOCKER_EMAIL)" -u="$(DOCKER_USERNAME)" -p="$(DOCKER_PASSWORD)"
	docker push $(REP_IMAGE)

tag-latest: rep-image2	# tag current REP image as latest
	docker tag -f $(REP_IMAGE) yandex/rep:latest

push-latest: tag-latest push	# tag current REP image as latest and push it to docker hub
	docker push yandex/rep:latest
