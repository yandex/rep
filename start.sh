#!/bin/bash

set -x

echo "Umask: "
umask
[ -z "$ENV_BIN_DIR" ] && source /etc/profile.d/rep_profile.sh

set -v
JUPYTER_CONFIG=$HOME/.jupyter/jupyter_notebook_config.py
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
	OPTIONS+=" --certfile=$SSL_CERTFILE"
fi

if [ "$SSL_KEYFILE" != "" ] ; then
	OPTIONS+=" --keyfile=$SSL_KEYFILE"
fi


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
	OPTIONS+=" --port $JUPYTER_PORT"
fi

[[ -d /REP_howto && ! -L /notebooks/rep_howto ]] && ln -s /REP_howto /notebooks/rep_howto

$HOME/install_modules.sh

echo "Starting Jupyter"
jupyter notebook $OPTIONS /notebooks 2>&1 | tee -a /notebooks/jupyter.log
