#!/bin/bash
# installing environment for REP with miniconda (not REP itself!)
# Usage:
# source install_rep_environment.sh PYTHON_MAJOR_VERSION
# where PYTHON_MAJOR_VERSION is 2 or 3.


# define a function to print error before exiting
function throw_error {
  echo -e $*
  exit 1
}

PYTHON_MAJOR_VERSION=2
if [ -n "$1" ]; then
    PYTHON_MAJOR_VERSION=$1
fi

# setting name (rep_py2 or rep_py3)
REP_ENV_NAME="rep_py${PYTHON_MAJOR_VERSION}"

# checking that system has apt-get
if which apt-get > /dev/null; then
    sudo apt-get update
    sudo apt-get install -y  \
        build-essential \
        libatlas-base-dev \
        liblapack-dev \
        libffi-dev \
        gfortran \
        git \
        libxft-dev \
        libxpm-dev \
        wget \
        telnet \
        curl
    # cleaning everything we can
    sudo apt-get clean
    sudo apt-get autoclean
    sudo apt-get autoremove
fi


# matplotlib and ROOT both using DISPLAY environment variable
# changing matplotlib configuration file to avoid conflict
mkdir -p $HOME/.config/matplotlib && echo 'backend: agg' > $HOME/.config/matplotlib/matplotlibrc

# exit existing environments
# TODO conda may not set CONDA_ENV_PATH
[ -n "$VIRTUAL_ENV" ] && deactivate
[ -n "$CONDA_ENV_PATH" ] && source deactivate

if ! which conda ; then
    echo "installing miniconda root environment with jupyterhub"
    MINICONDA_FILE="Miniconda3-3.19.0-Linux-x86_64.sh"
    wget http://repo.continuum.io/miniconda/$MINICONDA_FILE -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda || throw_error "Error installing miniconda"
    rm ./miniconda.sh
    export PATH=$HOME/miniconda/bin:$PATH
    hash -r
    conda update --yes conda
    pip install jupyterhub==0.6.1 notebook==4.0
    # cleaning root environment
    conda clean --yes --all

    [ -f $HOME/miniconda/bin/jupyterhub-singleuser ] || throw_error "jupyterhub inaccessible using standard path"
fi

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REP_ENV_FILE="$HERE/environment-rep${PYTHON_MAJOR_VERSION}.yaml"

echo "Creating conda venv $REP_ENV_NAME"
conda env create -q --file $REP_ENV_FILE > /dev/null
source activate $REP_ENV_NAME || throw_error "Error installing $REP_ENV_NAME environment"

echo "Removing conda packages and caches:"
conda uninstall --force --yes -q gcc qt
conda clean --yes --all
conda list


echo "Test installed packages:"
source $(which thisroot.sh) || throw_error "Error installing ROOT"
python -c 'import ROOT, root_numpy' || throw_error "Error installing root_numpy"
python -c 'import xgboost' || throw_error "Error installing XGBoost"

echo "Registering this environment as kernel for jupyterhub and jupyter"
python -m ipykernel.kernelspec

echo "Generating jupyter config"
jupyter notebook -y --generate-config

cat << EOL_CONFIG >> $HOME/.jupyter/jupyter_notebook_config.py
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
EOL_CONFIG

echo "Python version:"
which python
python --version

# creating correct profile (will be sourced in .bashrc)
mkdir -p /etc/profile.d/
cat >/etc/profile.d/rep_profile.sh << EOL_PROFILESH
    export PATH=$HOME/miniconda/bin:\$PATH
    source activate ${REP_ENV_NAME}
    # actually, ROOT shall be alredy sourced by previous command
    source $(which thisroot.sh) || echo "Could not source ROOT!"

EOL_PROFILESH

# printing message about environment
cat << EOL_MESSAGE

    # add to your environment (e.g. to $HOME/.bashrc file):
    source /etc/profile.d/rep_profile.sh
    # you also need to install rep with pip

EOL_MESSAGE
