FROM ubuntu:18.04

# Install apt tools
RUN apt update && \
    apt install -y --no-install-recommends \
          build-essential \
          ca-certificates \
          cmake \
          zip \
          unzip \
          vim \
          wget \
          curl \
          git \
          git-lfs \
          apt-transport-https \
          openssh-client \
          openssh-server \
          iputils-ping \
          net-tools \
          htop \
          bc \
          iproute2 \
          iperf && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install Python
RUN conda install -y python=3.9 && \
    conda update --all && \
    conda clean -ya

# Clone repo from github and install required packages
WORKDIR /root
RUN git lfs install && \
    git clone https://github.com/Lizonghang/TPI-LLM && \
    cd /root/TPI-LLM && \
    pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH /root/TPI-LLM/src:$PYTHONPATH

WORKDIR /root/TPI-LLM
CMD ["/bin/bash"]