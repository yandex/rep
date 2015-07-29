CERT=
PROFILE_CONFIG=/root/.ipython/profile_default/ipython_notebook_config.py
PASSWD_FILE=/root/.ipython/profile_default/nbpasswd.txt

if [ -n "$CERTFILE" ] ; then
  if [ -f $CERTFILE ] ; then
    echo "Certificate for the IPython: $CERTFILE"
    echo -e "c.NotebookApp.certfile = u'$CERTFILE'\n" >> $PROFILE_CONFIG
    CERT=$CERTFILE
  else
    echo "WARN: \$CERTFILE is specified ($CERTFILE), but cannot be found"
  fi
fi
if [[ -n "$CERTHOSTNAME" && -z "$CERTFILE" ]] ; then
  echo "Recreating certificate for $CERTHOSTNAME"
  CERT=/root/mycert.pem
  echo -e "\n\n\n\n${CERTHOSTNAME}\n\n" |
        openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout $CERT -out $CERT
fi

# INSTALL SECTION
[ -f "/etc_docker/requirements.txt" ] && pip install --upgrade -r /etc_docker/requirements.txt

# apt-get install -y telnet mc

# CONFIGURATION
umask 0002
echo umask 0002 >> /etc/bashrc

cat >> $PROFILE_CONFIG << EOF
# Notebook config
c.NotebookApp.certfile = u'$CERT'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False

c.NotebookApp.base_url = '/'
c.NotebookApp.webapp_settings = {'static_url_prefix':'/static/'}
EOF

if [ -n "$PASSWD" ] ; then
  python -c "from IPython.lib import passwd; print passwd('$PASSWD')"  > $PASSWD_FILE
  echo "c.NotebookApp.password = open('$PASSWD_FILE').read().strip()" >> $PROFILE_CONFIG
fi
