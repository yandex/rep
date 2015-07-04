FROM anaderi/rep-base:0.6
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>

ENV TEMP /tmp
RUN mkdir $TEMP/build
COPY setup.py README.md AUTHORS requirements.txt $TEMP/build/
COPY rep $TEMP/build/rep
COPY run.sh /root/
RUN cd $TEMP/build && \
  pip install . && \
  rm -rf $TEMP/build

# system setup
#
ENV PORT_IPYTHON=8080

WORKDIR /root/
RUN ipython profile create default &&\
  echo -e "\n\n\n\n`hostname`\n\n" | openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
\
  echo -e "# Notebook config\nc.NotebookApp.certfile = u'/root/mycert.pem'\nc.NotebookApp.ip = '*'\nc.NotebookApp.open_browser = False\n# It is a good idea to put it on a known, fixed port\nc.NotebookApp.port = $PORT_IPYTHON\n\nPWDFILE='/root/.ipython/profile_default/nbpasswd.txt'\nc.NotebookApp.password = open(PWDFILE).read().strip()" >> /root/.ipython/profile_default/ipython_notebook_config.py && \
  python -c "from IPython.lib import passwd; print passwd('rep')"  > /root/.ipython/profile_default/nbpasswd.txt

# COPY ipython-patch/custom.js /root/.ipython/profile_default/static/custom/custom.js

EXPOSE $PORT_IPYTHON
EXPOSE 5000
CMD ["bash", "/root/run.sh"]
