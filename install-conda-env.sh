#!/bin/bash

halt() {
  echo $*
  exit 1
}

[ -z "$HOME" ] && export HOME="/root"
apt-get update
apt-get install -y  \
    build-essential \
    libatlas-base-dev \
    liblapack-dev \
    libffi-dev \
    wget \
    gfortran \
    git \
    libxft-dev \
	libxpm-dev

mkdir $HOME/.config/matplotlib -p
echo 'backend: agg' > $HOME/.config/matplotlib/matplotlibrc
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b && rm ./miniconda.sh || halt "Error installing miniconda"
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
conda create --file environment_py2.yaml
conda uninstall --yes gcc qt
conda clean -p -t

source $HOME/miniconda/bin/thisroot.sh
[ -n "$ROOTSYS" ] || halt "Error installing ROOT"
python -c 'import ROOT, root_numpy' || halt "Error installing root_numpy"

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
python -c 'import xgboost' || halt "Error installing XGboost"

echo 'source activate py2' >> $HOME/.bashrc
echo 'source $ENV_BIN_DIR/thisroot.sh' >> $HOME/.bashrc
