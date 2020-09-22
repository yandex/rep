# Integration scripts for REP

Single pipeline is used in
 
1. testing on continuous integration systems
2. to deploy on machines 
3. and to cook docker images.

## Docker images

Docker image with REP is built using `Dockerfile.rep` 

In the main project folder run `make -e PYTHON=2 rep-image` or `make -e PYTHON=3 rep-image` 
    
## Installation process for ubuntu machines:

See `.travis.yml` (in the root folder), this code is used in testing, so it is guaranteed to work.