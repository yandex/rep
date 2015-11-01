#!/bin/bash

halt() {
  echo $*
  exit 1
}

apt-get update
apt-get install -y \
  python2.7 \
  python-dev \
  python-setuptools \
  libatlas-base-dev \
  gfortran \
  liblapack-dev

apt-get install --force-yes -y git dpkg-dev make g++ gcc binutils libx11-dev \
        libxpm-dev libxft-dev libxext-dev libpng3 libjpeg8 libssl-dev \
        libpcre3-dev libgl1-mesa-dev libglew1.5-dev libftgl-dev \
        libmysqlclient-dev libfftw3-dev libcfitsio3-dev graphviz-dev \
        libavahi-compat-libdnssd-dev libldap2-dev libxml2-dev \
        libkrb5-dev libgsl0-dev libqt4-dev libffi-dev libncurses5-dev \
        graphviz

apt-get install --force-yes -y curl python-pycurl libcurl4-gnutls-dev wget \
        python-numpy

easy_install pip

pip install -r https://raw.githubusercontent.com/yandex/rep/develop/requirements.txt || halt "Error installing REP reqs"

#
# XGboost
#

export TEMP='/tmp'
pip install graphviz
git clone https://github.com/dmlc/xgboost.git $TEMP/xgboost && \
  cd $TEMP/xgboost

git checkout 8e4dc4336849c24ae48636ae60f5faddbb789038 && \
  ./build.sh && \
  cd python-package && \
  python setup.py install && \
  cd ..

PYTHONPATH+=:tests/python python -m unittest  test_basic.TestBasic || halt "Error installing XGboost"

cd / && rm -rf $TEMP/xgboost

#
# root.cern.ch
#
apt-get install -y root-system libroot-bindings-python-dev \
        libroot-roofit5.34 libroot-roofit-dev
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/root5.34/:$LD_LIBRARY_PATH"
export PYTHONPATH="/usr/lib/x86_64-linux-gnu/root5.34:$PYTHONPATH"
 
python -c 'import ROOT' || halt "Error installing ROOT"

pip install \
  rootpy==0.8.0 \
  root_numpy==4.3.0

python -c 'import root_numpy' || halt "Error installing root_numpy"


