#!/bin/bash
apt-get update
apt-get install -y  \
    build-essential \
    libatlas-base-dev \
    liblapack-dev \
    libffi-dev \
    wget \
    gfortran \
    git


mkdir $HOME/.config/matplotlib -p
echo 'backend: agg' > $HOME/.config/matplotlib/matplotlibrc
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
rm ./miniconda.sh
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
conda install --yes python=2.7
conda install --yes -q -c remenska root_numpy
source $HOME/miniconda/bin/thisroot.sh
echo $ROOTSYS
conda install --yes pip numpy scipy six pandas setuptools
conda install --yes theano scikit-learn matplotlib ipython-notebook
conda uninstall --yes gcc
 # install xgboost
git clone https://github.com/dmlc/xgboost.git
cd xgboost
 # taking particular xgboost commit, which is working
git checkout 8e4dc4336849c24ae48636ae60f5faddbb789038
./build.sh
cd python-package
python setup.py install
cd ../..
 # end install xgboost
python -c 'import xgboost'
python -c 'import ROOT'
python -c 'import root_numpy'
pip install -q neurolab==0.3.5 theanets==0.6.2 pybrain==0.3

