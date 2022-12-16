FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116
ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 
RUN apt update -y && apt install -y sudo
RUN groupadd --gid $USER_GID $USERNAME &&\
 useradd --uid $USER_UID --gid $USER_GID -m $USERNAME &&\  
 echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} &&\
 chmod 0440 /etc/sudoers.d/${USERNAME}

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=~/.bash_history" && echo $SNIPPET >> "/home/${USERNAME}/.bashrc"

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt install -y \
    build-essential \
    curl \
    git

# GPU Setup
RUN apt-get install -y \
    libcairo2-dev \
    libgl1-mesa-glx \
    software-properties-common

# Install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9-dev python3.9-venv
RUN python3.9 -m ensurepip
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python
RUN ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip
RUN pip install --upgrade pip

USER ${USERNAME}

CMD mkdir -p /code
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN git config --global credential.helper store
RUN /home/dev/.local/bin/jupyter contrib nbextension install --user
RUN sudo apt install jq -y
RUN /home/dev/.local/bin/jupyter nbextension enable gist_it/main
ADD . .