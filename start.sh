#!/bin/bash

set -v
JUPYTER_CONFIG=$HOME/.jupyter/jupyter_notebook_config.py
if [ "$JPY_API_TOKEN" != "" ] ; then
	echo "Starting under Jupyterhub"
	source activate jupyterhub_py3
	jupyterhub-singleuser $*
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
	echo "Setting up password support for Jupyther profile"
	sha=`python -c "from notebook.auth import passwd; print passwd('$PASSWORD')"`
	echo "c.NotebookApp.password = u'$sha'" >> $JUPYTER_CONFIG
fi

if [ "$SECRET" != ""] ; then
	echo "c.NotebookNotary.secret = b'$SECRET'" >> $JUPYTER_CONFIG
fi

if [ "$SECRET_FILE" != "" ] ; then
	echo "c.NotebookNotary.secret_file = '$SECRET_FILE'" >> $JUPYTER_CONFIG
fi

if [ "$JUPYTER_PORT" != "" ] ; then
	echo "New Jupyter port is $JUPYTER_PORT"
	OPTIONS+=" --port $JUPYTER_PORT"
fi

[[ -d /REP_howto && ! -L /notebooks/rep_howto ]] && ln -s /REP_howto /notebooks/rep_howto

echo "Starting Jupyter"
jupyter notebook $OPTIONS