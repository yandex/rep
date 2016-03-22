
# Running ipyparallel cluster with REP & docker swarm


## Setup

prerequisites:
- Kernel 3.10+
- docker 1.8+ on all machines
- list all your nodes in `cluster.txt` (just one per line, like `cluster.txt.orig`)

## Start

`make start-master`
`make start-slaves` will start slaves
to test
`make test-cluster`

jupyter with REP will be acccessible by port 8888 of node hosting master instance.


## Stop

	make stop-cluster 

## Maintenance

to add more slaves:

```make -e N=5 start-slaves``` -- start 5 more slaves

## Troubleshooting
