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
export PATH+=:$HOME/miniconda/bin
./miniconda.sh -b && rm ./miniconda.sh || halt "Error installing miniconda"
# conda update --yes conda
conda env create --file environment_py2.yaml || halt "Error installing py2 environment"
source activate py2
conda uninstall --yes gcc qt
conda clean --yes -p -t

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

# test installed packages
source $ENV_BIN_DIR/thisroot.sh || halt "Error installing ROOT"
python -c 'import ROOT, root_numpy' || halt "Error installing root_numpy"
python -c 'import xgboost' || halt "Error installing XGboost"

# environment
cat << EOF > $HOME/.bashrc
export PATH=/root/miniconda/bin:$PATH
source activate py2
source $ENV_BIN_DIR/thisroot.sh
EOF
