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
WORKDIR /root/
RUN ipython profile create
# COPY ipython-patch/custom.js /root/.ipython/profile_default/static/custom/custom.js

EXPOSE 8080
EXPOSE 5000
CMD ["bash", "run.sh"]
