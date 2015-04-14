# Reproducible Experiment Platform (REP)

REP is environment for conducting data-driven research in a consistent and reproducible way.

Main REP features include:

  * unified classifiers wrapper for variety of implementations (TMVA, Sklearn, XGBoost, Uboost)
  * parallel training of classifiers on cluster 
  * classification/regression reports with plots
  * support of interactive plots
  * grid-search with parallelized execution on a cluster
  * git, versioning of research
  * computation of different classification metrics 


## Requirements

        'numpy >= 1.9.0',
        'scipy >= 0.14.0',
        'ipython[all] == 3.0.0',
        'pyzmq == 14.3.1',
        'matplotlib == 1.3.1',
        'openpyxl < 2.0.0',
        'pandas == 0.14.0',
        'requests >= 2.5.1',
        'Jinja2 >= 2.7.3',
        'numexpr >= 2.4',
        'plotly == 1.2.3',
        'scikit-learn == 0.15.2',
        'bokeh == 0.8.1',
        'mpld3 == 0.2',

## Install

### Preliminary for Ubuntu

    sudo apt-get update

For clear ubuntu system:

    sudo apt-get install python-dev libblas-dev libatlas-dev liblapack-dev gfortran g++

For matplotlib:

    sudo apt-get install libpng-dev libjpeg8-dev libfreetype6-dev

Update pip:

    sudo pip install --upgrade pip


### REP Installation

After cloning and (optionally) setting up virtual environment run:

    export LC_ALL=C
    pip install .


To use uBoost and other uniforming classifiers:

1. clone https://github.com/anaderi/lhcb_trigger_ml
2. install it (pip install -e .)


To use TMVA library:

1. install ROOT 5.XX.XX https://root.cern.ch/drupal/content/installing-root-source
2. add to bashrc

   source PATH_TO_ROOT/libexec/thisroot.sh

3. Install additional libraries to work with ROOT

        pip install root_numpy==3.3.1
        pip install rootpy==0.7.1



To use XGBoost library:

1. clone https://github.com/tqchen/xgboost
2. install it (make), add /wrapper folder to PYTHONPATH (i.e. by modifying easy-install.pth)


### Run

    mkdir notebook
    cd notebook
    ipython notebook

Go to [http://localhost:8888](http://localhost:8888) (in case of running on a localhost)

For server instead of the latest command use:

    ipython notebook --ip="*"

Open [http://server:8888](http://server:8888)


Explanations:

* 8080 port - for ipython notebook
* 5000 port - for nbdiff
* /notebooks - folder with notebooks (data should be here as well)
* /logdir - folder for ipython notebook logs

To play with examples, copy `howto` folder from this repository to `$HOME/notebooks`   