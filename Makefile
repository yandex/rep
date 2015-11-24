# Makefile is used for building & starting rep-containers
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
CONTAINER_NAME:=$(shell basename $(CURDIR) | tr - _ )

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

run:
	docker run -ti --rm -p 8888:8888 --name $(CONTAINER_NAME) $(REP_IMAGE) $(RUN_ARGS) 

restart:
	docker restart $(CONTAINER_NAME)

run-daemon:
	docker run -d --name $(CONTAINER_NAME) $(REP_IMAGE) $(RUN_ARGS) 

exec:
	docker exec $(CONTAINER_NAME) $(EXEC_ARGS)

logs:
	docker logs $(CONTAINER_NAME)

stop:
	docker stop $(CONTAINER_NAME)

remove: stop
	docker rm $(CONTAINER_NAME)

inspect:
	docker inspect $(REP_IMAGE) 

push: rep-image
        docker login -e="$(DOCKER_EMAIL)" -u="$(DOCKER_USERNAME)" -p="$(DOCKER_PASSWORD)"
	docker push $(REP_IMAGE)

push-base:
	docker push $(REP_BASE_IMAGE)

tag-latest: rep-image
	docker tag -f $(REP_IMAGE) yandex/rep:latest

push-latest: tag-latest push
	docker push yandex/rep:latest
