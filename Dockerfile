FROM ubuntu:14.04
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>
ADD ./install-base.sh /tmp/
RUN /tmp/install-base.sh

ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/root5.34/:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/usr/lib/x86_64-linux-gnu/root5.34:$PYTHONPATH"


CMD ["bash"]
