FROM yandex/rep:0.6.5
RUN bash --login -c "pip install ipyparallel==4.1.0"
ENV HOME /root
COPY start_jupyter.sh $HOME/start_jupyter.sh

CMD ["/bin/bash", "--login", "-c", "$HOME/start_jupyter.sh"]
