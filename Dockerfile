#-------------------------------------------------------------------
FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#-------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    unzip \
    cmake \ 
    curl \
    tmux \
    htop \ 
    vim \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \ 
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
#-------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash teleuser
WORKDIR /home/teleuser
USER teleuser

#-------------------------------------------------------------------
ENV MCND https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
ENV PATH /home/teleuser/mambaforge/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
RUN curl -L "$MCND" > mambaforge.sh && \
    bash mambaforge.sh -bfp /home/teleuser/mambaforge
#-------------------------------------------------------------------
EXPOSE 6006 8888
WORKDIR /ec2/
CMD ["/bin/bash"]
#-------------------------------------------------------------------