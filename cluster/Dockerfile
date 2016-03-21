FROM yandex/rep:0.6.5
ENV HOME /root
RUN bash --login -c "pip install ipyparallel==4.1.0"
COPY start_jupyter.sh /root/start_jupyter.sh

CMD ["/bin/bash", "--login", "-c", "$HOME/start_jupyter.sh"]
