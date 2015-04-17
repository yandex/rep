FROM ubuntu:14.04
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>
#MAINTAINER Egor Khairullin <mikari@yandex-team.ru>
RUN apt-get update && apt-get install -y \
  python2.7 \
  python-pip \
  python-dev \
  libatlas-base-dev \
  gfortran \
  liblapack-dev

RUN apt-get install git dpkg-dev make g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev libpng3 libjpeg8 gfortran libssl-dev libpcre3-dev libgl1-mesa-dev libglew1.5-dev libftgl-dev libmysqlclient-dev libfftw3-dev libcfitsio3-dev graphviz-dev libavahi-compat-libdnssd-dev libldap2-dev python-dev libxml2-dev libkrb5-dev libgsl0-dev libqt4-dev curl python-pycurl --force-yes -y

RUN apt-get install libcurl4-gnutls-dev -y --force-yes

RUN easy_install distribute
RUN pip install cython
RUN pip install "ipython[notebook]"
RUN pip install \
  numpy==1.9.1 \
  pandas==0.14.0

RUN pip install \
  pyzmq==14.3.0 \
  pycurl==7.19.3 \
  scipy==0.14.0 \
  plotly \
  numexpr==2.4 \
  jinja2==2.7.3

RUN pip install scikit-learn==0.15.2
RUN pip install matplotlib==1.3.1



WORKDIR /root
RUN git clone http://root.cern.ch/git/root.git \
  && cd root \
  && git checkout v5-34-21 \
  && ./configure --prefix=/usr/local \
  && make -j2 \
  && make install \
  && cd .. \
  && rm -rf root


ENV ROOTSYS /usr/local
ENV PATH /usr/local/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/lib/root:$LD_LIBRARY_PATH
ENV PYTHONPATH /usr/local/lib/root

RUN pip install \
  rootpy==0.7.1 \
  root_numpy==3.3.1

RUN git clone https://github.com/tqchen/xgboost.git /xgboost
WORKDIR /xgboost
RUN make

ENV PYTHONPATH /usr/local/lib/root:/xgboost/wrapper

RUN apt-get install ruby ruby-dev -y --force-yes
RUN gem install nokogiri -v '1.6.3.1'
WORKDIR /root/
RUN git clone https://github.com/alebedev/git-media.git
WORKDIR /root/git-media
RUN gem install bundler
RUN bundle install
RUN gem build git-media.gemspec
RUN gem install git-media-*.gem
WORKDIR /root/
RUN rm -rf /root/git-media

RUN cat /etc/resolv.conf && ifconfig
RUN echo nameserver 8.8.8.8 >> /etc/resolv.conf && cat /etc/resolv.conf && ifconfig
RUN ping -c  1 ya.ru
WORKDIR /root/
RUN git clone https://github.com/tarmstrong/nbdiff.git
RUN cd /root/nbdiff && pip install . && cd /root && rm -rf nbdiff

RUN pip install pyyaml xlrd

RUN mkdir /root/build

COPY setup.py README.md AUTHORS requirements.txt /root/build/
COPY rep /root/build/rep
COPY scripts/docker_start.sh /root/start.sh
RUN ls -lR /root
RUN cd /root/build && \
  pip install . && \
  rm -rf /root/build

WORKDIR /root/
RUN ipython profile create
# COPY ipython-patch/custom.js /root/.ipython/profile_default/static/custom/custom.js

EXPOSE 8080
EXPOSE 5000
CMD ["sh", "start.sh"]
