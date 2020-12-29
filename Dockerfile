FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV LANG=C.UTF-8
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openssh-server  unzip curl \
    libx11-dev  libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common xpra xserver-xorg-dev \
    cmake libopenmpi-dev python3-dev zlib1g-dev gcc g++ \
    iputils-ping net-tools  iproute2  htop xauth \
    tmux wget vim git bzip2 ca-certificates  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    sed -i 's/^#X11UseLocalhost.*$/X11UseLocalhost no/' /etc/ssh/sshd_config && \
    sed -i 's/^#AddressF.*$/AddressFamily inet/' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd && \
    echo 'root:teslazhu' | chpasswd && \
    sed -i 's/^.*PermitRootLogin.*$/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile && \
    echo "conda activate base" >> /etc/profile

WORKDIR /root/code
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U  && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 6000

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MJPRO_PATH /root/.mujoco/mjpro150
ENV MUJOCO_PY_MJKEY_PATH /root/.mujoco/mjkey.txt
ENV envname torch

RUN . /opt/conda/etc/profile.d/conda.sh  && \
    conda create -y -n $envname python==3.6.7 && \
    sed -i 's/conda activate base/conda activate '"$envname"'/g' /etc/profile && \
    conda activate $envname && \
    git clone https://github.com/openai/gym.git \
    && cd gym \
    && git checkout tags/0.15.7 \
    && conda install -y lockfile mpi4py patchelf pyyaml \
    && pip install .[atari,mujoco] \
    && cd .. \
    && conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch  \
    && conda install -y tensorboard \
    && conda clean -ay \
    && pip cache purge

ENV PATH /opt/conda/envs/${envname}/bin:$PATH
EXPOSE 6006

RUN echo 'export LD_LIBRARY_PATH=/root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH' >> /etc/profile && \
    echo "export LANG=C.UTF-8" >>  /etc/profile && \
    echo "MUJOCO_PY_MJPRO_PATH=/root/.mujoco/mjpro150" >>  /etc/profile && \
    echo "MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt" >>  /etc/profile && \
    mkdir -p /root/.ssh
COPY id_rsa /root/.ssh
RUN chmod 600 /root/.ssh/id_rsa

ENV HOST_UID 1024
ENV HOST_GID 1024
ENV USER_NAME teslazhu
RUN addgroup --gid ${HOST_GID} ${USER_NAME} && \
    adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    --gid ${HOST_GID} \
    ${USER_NAME} && \
    echo ${USER_NAME}:${USER_NAME} | chpasswd && \
    chmod o+w -R /opt/conda/envs/torch/lib/python3.6/site-packages/mujoco_py/generated/ && \
    chmod o+x /root && \
    chmod o+rwx -R /root/.mujoco
CMD ["sleep", "100d"]
USER ${HOST_UID}:${HOST_GID}
