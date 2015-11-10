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
REPTAG=0.6.4
REPBASETAG=0.6.1


.PHONY: run rep-image rep-base-image inspect

rep-base-image:
	docker build -t yandex/rep-base:$(REPBASETAG) -f ci/Dockerfile.base ci

rep-image:
	docker build -t yandex/rep:$(REPTAG) -f ci/Dockerfile.rep .

run:
	docker run -ti --rm -p 8888:8888 --name $(CONTAINER_NAME) yandex/rep:$(REPTAG) $(RUN_ARGS) 

restart:
	docker restart $(CONTAINER_NAME)

run-daemon:
	docker run -d --name $(CONTAINER_NAME) yandex/rep:$(REPTAG) $(RUN_ARGS) 

exec:
	docker exec $(CONTAINER_NAME) $(EXEC_ARGS)

stop:
	docker stop $(CONTAINER_NAME)

remove: stop
	docker rm $(CONTAINER_NAME)

inspect:
	docker inspect yandex/rep:$(REPTAG) 

push:
	docker push yandex/rep:$(REPTAG)

tag-latest: rep-image
	docker tag -f yandex/rep:$(REPTAG) yandex/rep:latest

push-latest: push tag-latest
	docker push yandex/rep:latest
