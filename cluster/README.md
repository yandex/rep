
# Running ipyparallel cluster with REP & docker swarm


## Setup

prerequisites:
- Kernel 3.10+
- docker 1.8+ on all machines
- list all your nodes in `cluster.txt` (just one per line, like `cluster.txt.orig`)

## Start

`make start-master` -- start master

`make start-slaves` -- start slaves (by number of lines in `cluster.txt`)

jupyter with REP will be acccessible by port 8888 of master instance.


## Stop

	make stop-cluster 

## Maintenance


to check cluster status `make test-cluster`

to add, say, 5 more slaves: ```make -e N=5 start-slaves```



## Troubleshooting
