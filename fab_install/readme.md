# Installing with Fabric

Fabric is a command line tool for automatic deployment and relevant tasks. Fabric scenario is suggested to install REP on a new server.
   
Currently only one installation script is suggested, it set ups REP on Ubuntu 14.04. The script has been tested on Digital Ocean droplet however it should work with any Ubuntu 14.04 installation.

## How to use

Before using the script you should install fabric and fabtools:

`pip install fabric fabtools`

Clone `scripts` branch:

`git clone -b scripts --single-branch https://github.com/yandex/rep.git` 

Go to fab_install directory 

`cd rep/fab_install`

and slightly modify the file: change the lines in the very beginning of the file with your actual servers information.

`nano fabfile.py`

Finally run fabric:

`fab setup_rep`