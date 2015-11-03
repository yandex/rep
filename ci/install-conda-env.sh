#!/bin/bash

set -v

halt() {
  echo $*
  exit 1
}

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYSTEM=`uname -s`

[ -z "$HOME" ] && export HOME="/root"
if [ $SYSTEM == "Linux" ] && which apt-get > /dev/null ; then
    sudo apt-get update
    sudo apt-get install -y  \
        build-essential \
        libatlas-base-dev \
        liblapack-dev \
        libffi-dev \
        wget \
        gfortran \
        git \
        libxft-dev \
        libxpm-dev
fi

mkdir -p $HOME/.config/matplotlib
echo 'backend: agg' > $HOME/.config/matplotlib/matplotlibrc
if [ -n "$TRAVIS_PYTHON_VERSION" ] ; then
    PENV_NAME="py${TRAVIS_PYTHON_VERSION:0:1}"
else
    PVERSION=`python --version 2>&1|awk '{print $2}'`
    PENV_NAME="py${PVERSION:0:1}"
fi
if ! which conda ; then
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b && rm ./miniconda.sh || halt "Error installing miniconda"
    export PATH=$HOME/miniconda/bin:$PATH
fi
ENV_FILE=$HERE/environment.yaml
[ -f $HERE/environment_${SYSTEM}.yaml ] && ENV_FILE=environment_${SYSTEM}.yaml
conda env create --name $PENV_NAME --file $ENV_FILE #|| halt "Error installing $PENV_NAME environment"
source activate $PENV_NAME
conda uninstall --yes gcc qt
conda clean --yes -pt

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
cat << EOF
# add to your environment:
export PATH=$HOME/miniconda/bin:$PATH
source activate $PENV_NAME
pushd $ENV_BIN_DIR/.. ; source 'bin/thisroot.sh' ; popd
EOF
