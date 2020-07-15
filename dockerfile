FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN export TERM=xterm-256color
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)"
RUN chsh -s `which zsh`

RUN git clone https://github.com/0xDECAFC0FFEE/.setup.git /root/.setup
RUN rm /root/.zshrc
RUN python3 /root/.setup/zshrc_src/init_zshrc.py
CMD `which zsh`

RUN pip3 install -f /workspace/STAMP2/requirements.txt
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN apt-get -y install tmux
