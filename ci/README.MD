# Integration scripts for REP

Single pipeline is used in
 
1. testing on continuous integration systems
2. to deploy on machines 
3. and to cook docker images.

## Docker images

Docker image with REP is built in two steps: 

1. `docker-base` is built with all dependencies (including ROOT).
    - `Dockerfile.repbase` is used (see the code).
    - it installs conda environments with all packages
 
2. `docker-rep` is built on the top of `docker-base`.
    - `Dockerfile.rep` is used.
    - Each minor release changes this container, not `repbase` container.
    
## Installation process for ubuntu machines:

See `.travis.yml` (in the root folder), this code is used in testing, so it is guaranteed to work.