FROM ubuntu:14.04
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>, Alex Rogozhnikov <axelr@yandex-team.ru>

# This variable is set during building docker container
ARG REP_PYTHON_VERSION=2

# Setting environment variables in the container
ENV HOME=/root  \
    TEMP=/tmp  \
    PORT_JUPYTER=8888  \
    SHELL=/bin/bash

# Creating folders in container
RUN mkdir -p $TEMP/build \
 && mkdir -p /notebooks \
 && mkdir -p /REP_howto

# Copy REP to build folder
COPY ./ $TEMP/build/
# Copy HowTos separately
COPY ./howto /REP_howto

# changing default shell to bash. See http://stackoverflow.com/questions/20635472/
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# installing environment,
# adding profiles to .bashrc,
# pip-installing rep itself
# deleting folder
RUN source $TEMP/build/ci/install_rep_environment.sh $REP_PYTHON_VERSION \
 && echo "source /etc/profile.d/rep_profile.sh" >> $HOME/.bashrc \
 && echo "umask 0002" >> $HOME/.bashrc \
 && pip install $TEMP/build \
 && rm -rf $TEMP/build

# registering mounting points
VOLUME ["/notebooks"]
# registering port
EXPOSE $PORT_JUPYTER

# adding a file to run jupyter
COPY ./ci/start_jupyter.sh $HOME/

# starting IPython process when image is started
CMD ["/bin/bash", "--login", "-c", "$HOME/start_jupyter.sh"]
