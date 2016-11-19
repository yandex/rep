#!/bin/bash

# script to start jupyter inside docker container
# fine-tuned by set of environment variables (see the code),
# e.g. runs under jupyterhub environment in case JPY_API_TOKEN is set

set +xv

[ -z "$ENV_BIN_DIR" ] && source /etc/profile.d/rep_profile.sh

if [ "$INSTALL_PIP_MODULES" != "" ] ; then
	pip install $INSTALL_PIP_MODULES
fi

if [ "$JPY_API_TOKEN" != "" ] ; then
	echo "Starting under Jupyterhub"
	jupyter kernelspec install-self
	source activate jupyterhub_py3
	jupyter kernelspec install-self
	source activate rep_py2 # default env

	NOTEBOOK_DIR=/notebooks
	git clone $JPY_GITHUBURL $NOTEBOOK_DIR
	$HOME/miniconda/envs/jupyterhub_py3/bin/jupyterhub-singleuser \
	  --port=8888 \
	  --ip=0.0.0.0 \
	  --user=$JPY_USER \
	  --cookie-name=$JPY_COOKIE_NAME \
	  --base-url=$JPY_BASE_URL \
	  --hub-prefix=$JPY_HUB_PREFIX \
	  --hub-api-url=$JPY_HUB_API_URL \
	  --notebook-dir=$NOTEBOOK_DIR
	exit $?
fi

if [ "$GENERATE_SSL_HOSTNAME" != "" ] ; then
	echo "Setting up SSL support for the Jupyter profile"
	SSL_CERTFILE="/root/mycert.pem"
	SSL_KEYFILE=""
  	echo -e "\n\n\n\n${GENERATE_SSL_HOSTNAME}\n\n" |
        openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout $SSL_CERTFILE -out $SSL_CERTFILE
fi

if [ "$SSL_CERTFILE" != "" ] ; then
	JUPYTER_OPTIONS+=" --certfile=$SSL_CERTFILE"
fi

if [ "$SSL_KEYFILE" != "" ] ; then
	JUPYTER_OPTIONS+=" --keyfile=$SSL_KEYFILE"
fi

JUPYTER_CONFIG=$HOME/.jupyter/jupyter_notebook_config.py

if [ "$PASSWORD" != "" ] ; then
	sha=`python -c "from notebook.auth import passwd; print passwd('$PASSWORD')"`
	echo "c.NotebookApp.password = u'$sha'" >> $JUPYTER_CONFIG
fi

if [ "$SECRET" != "" ] ; then
	echo "c.NotebookNotary.secret = b'$SECRET'" >> $JUPYTER_CONFIG
fi

if [ "$SECRET_FILE" != "" ] ; then
	echo "c.NotebookNotary.secret_file = '$SECRET_FILE'" >> $JUPYTER_CONFIG
fi

if [ "$JUPYTER_PORT" != "" ] ; then
	JUPYTER_OPTIONS+=" --port $JUPYTER_PORT"
fi

if [ "$IPPHUB_IP" != "" ] ; then
        ipython profile create --parallel
        ipcontroller --ip=* --port 32000 --location=$IPPHUB_IP --HubFactory.iopub=32001,32002 --HubFactory.hb=32003,32004 --HubFactory.mux=32005,32006 --HubFactory.notifier_port=32007 --HubFactory.task=32008,32009 --HubFactory.control=32010,32011 &
fi

[[ -d /REP_howto && ! -L /notebooks/rep_howto ]] && ln -s /REP_howto /notebooks/rep_howto

cat .rep_version
source .rep_version
echo "Starting Jupyter"
jupyter notebook $JUPYTER_OPTIONS /notebooks 2>&1 | tee -a /notebooks/jupyter.log
