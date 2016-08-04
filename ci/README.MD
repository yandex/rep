# Integration scripts for REP

Single pipeline is used in
 
1. testing on continuous integration systems
2. to deploy on machines 
3. and to cook docker images.

## Docker images

Docker image with REP is built using `Dockerfile.rep` 

In the main project folder run `make rep-image2` or `make rep-image3` 
    
## Installation process for ubuntu machines:

See `.travis.yml` (in the root folder), this code is used in testing, so it is guaranteed to work.