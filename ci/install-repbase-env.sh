#!/bin/bash

set -v

halt() {
  echo -e $*
  exit 1
}

PYTHON_MAJOR_VERSION=2
[ "$1" == "-h" ] && halt "Usage: $0 [PYTHON_MAJOR_VERSION=2]\ne.g.: $0 3"
[ -n "$1" ] && PYTHON_MAJOR_VERSION=$1 && shift
if [ -n "$TRAVIS_PYTHON_VERSION" ] ; then
    PYTHON_MAJOR_VERSION=${TRAVIS_PYTHON_VERSION:0:1}
fi
REP_ENV_NAME="rep_py${PYTHON_MAJOR_VERSION}"

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYSTEM=`uname -s`

[ -z "$HOME" ] && export HOME="/root"
if [ $SYSTEM == "Linux" ] && which apt-get > /dev/null && ! dpkg -l |grep libxpm-dev > /dev/null ; then
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

mkdir -p $HOME/.config/matplotlib && 
  echo 'backend: agg' > $HOME/.config/matplotlib/matplotlibrc

# exit existing env
[ -n "$VIRTUAL_ENV" ] && deactivate
[ -n "$CONDA_ENV_PATH" ] && source deactivate

if ! which conda ; then
    MINICONDA_FILE="Miniconda-latest-Linux-x86_64.sh"
    [ "$PYTHON_MAJOR_VERSION" == "3" ] && MINICONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
    wget http://repo.continuum.io/miniconda/$MINICONDA_FILE -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda || halt "Error installing miniconda"
    rm ./miniconda.sh
    export PATH=$HOME/miniconda/bin:$PATH
    hash -r
    conda update --yes conda
fi

system_speficifc_env() {
    NAME=$1
    SYSTEM=$2
    [ -f "${NAME/%.yaml/_${SYSTEM}.yaml}" ] && NAME="${NAME/%.yaml/_${SYSTEM}.yaml}"
    echo $NAME
}

REP_ENV_FILE=`system_speficifc_env $HERE/environment-rep.yaml $SYSTEM`
JUPYTERHUB_ENV_FILE=`system_speficifc_env $HERE/environment-jupyterhub.yaml $SYSTEM`
conda env create --name $REP_ENV_NAME --file $REP_ENV_FILE 
conda env create --name jupyterhub_py3 --file $JUPYTERHUB_ENV_FILE
source activate $REP_ENV_NAME || halt "Error installing $REP_ENV_NAME environment"
conda uninstall --yes gcc qt
conda clean --yes -s -p -l -i -t

# install xgboost
git clone https://github.com/dmlc/xgboost.git
pushd xgboost
# taking particular xgboost commit, which is working
git checkout 8e4dc4336849c24ae48636ae60f5faddbb789038
./build.sh
cd python-package
python setup.py install
popd
rm -rf xgboost
# end install xgboost

# test installed packages
pushd $ENV_BIN_DIR/.. 
source 'bin/thisroot.sh' || halt "Error installing ROOT"
popd
python -c 'import ROOT, root_numpy' || halt "Error installing root_numpy"
python -c 'import xgboost' || halt "Error installing XGboost"
# environment
cat << EOF
# add to your environment:
export PATH=$HOME/miniconda/bin:$PATH
source activate $REP_ENV_NAME
pushd $ENV_BIN_DIR/.. ; source 'bin/thisroot.sh' ; popd
EOF
