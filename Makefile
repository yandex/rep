# Makefile for building & starting REP containers
# arguments can be supplied by -e definitions: 
#
#	NOTEBOOKS -- folder to be mounted to container (default: ./notebooks)
#	PORT -- port to listen for incoming connection (default: 8888)
#	PYTHON -- python version to use in image (2 or 3, default 2)
#

PYTHON ?= 2
NOTEBOOKS ?= $(shell pwd)/notebooks
PORT ?= 8888
DOCKER_ARGS := --volume $(NOTEBOOKS):/notebooks -p $(PORT):8888

HERE := $(shell pwd)
REP_IMAGE_NAME_PY2 := yandex/rep:0.6.6
REP_IMAGE_NAME_PY3 := $(REP_IMAGE_NAME_PY2)_py3

ifeq ($(PYTHON), 2)
	REP_IMAGE_NAME := $(REP_IMAGE_NAME_PY2)
	CONTAINER_NAME := rep_py2
else ifeq ($(PYTHON), 3)
	REP_IMAGE_NAME := $(REP_IMAGE_NAME_PY3)
	CONTAINER_NAME := rep_py3
else
	ERR := $(error Unknown python version)
endif



help:
	@echo Usage: make [-e VARIABLE=VALUE] targets
	@echo "variables:"
	@grep -h "#\s\+\w\+ -- " $(MAKEFILE_LIST) | sed "s/#\s//"
	@echo
	@echo targets:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' -e 's/^/   /' | sed -e 's/##//'

.PHONY: run rep-image run-daemon restart logs  \
	inspect exec help stop remove push push-base tag-latest2 push-latest2

version:
	@echo $(REP_IMAGE_NAME)

rep-image:	## build REP image with python set by PYTHON
	@echo "\n\nBuilding docker for python=$(PYTHON) \n\n"
	docker build --build-arg REP_PYTHON_VERSION=$(PYTHON) -t $(REP_IMAGE_NAME) -f ci/Dockerfile.rep .

local-dirs: # creates a local directory to be mounted to REP container
	[ -d $(NOTEBOOKS) ] || mkdir -p $(NOTEBOOKS)

run: local-dirs		## run REP interactively
	docker run --interactive --tty --rm $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE_NAME)

run-daemon: local-dirs	## run REP as a daemon
	docker run --detach $(DOCKER_ARGS) --name $(CONTAINER_NAME) $(REP_IMAGE_NAME)

run-tests:  ## run tests inside a container, both notebooks and scripts. Notebooks work only on python2!
	find tests -name '*.pyc' -delete
	# for some reason nosetests fails if directly mounted to tests folder
	mkdir -p $(HERE)/_docker_tests/
	cp -r $(HERE)/tests $(HERE)/_docker_tests/
	cp -r $(HERE)/howto $(HERE)/_docker_tests/
	docker run  --interactive --tty --rm --volume $(HERE)/_docker_tests:/notebooks $(REP_IMAGE_NAME) \
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

push: rep-image	# build REP image & push to docker hub
	# next line is @echoed in order not to show credentials during publishing
	@docker login -e="$(DOCKER_EMAIL)" -u="$(DOCKER_USERNAME)" -p="$(DOCKER_PASSWORD)"
	docker push $(REP_IMAGE_NAME)

tag-latest2: rep-image	# tag current REP image as latest
	docker tag -f $(REP_IMAGE_NAME_PY2) yandex/rep:latest

push-latest2: tag-latest2 push2	# tag current REP image as latest and push it to docker hub
	docker push yandex/rep:latest
