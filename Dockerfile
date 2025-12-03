FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Update packages, install build dependencies, and the specified version of CMake
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget make git gfortran libomp-18-dev libboost-all-dev clang-18 clang-tools-18 unzip && \
    apt-get install -y imagemagick vim git-lfs && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-linux-x86_64.sh && \
    ./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.0-linux-x86_64.sh

# setup alternative names for the compiler
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-18 100

# clone the repo into the container
RUN git clone git@github.com:Scientific-Computing-Lab/gpuFLOPBench.git /gpu-flopbench

# get the LFS files
RUN cd /gpu-flopbench && git checkout camera-ready && git pull && git lfs pull && git lfs fetch --all && git lfs checkout

# Set the working directory
WORKDIR /gpu-flopbench

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/anaconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

RUN source ~/anaconda3/bin/activate && \
    conda init --all && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN source ~/anaconda3/bin/activate && \
    conda create --name "gpu-flopbench" python=3.11.11 && \
    conda activate gpu-flopbench

RUN source ~/anaconda3/bin/activate && \
    conda activate gpu-flopbench && \
    conda install --channel conda-forge pygraphviz

RUN source ~/anaconda3/bin/activate && \
    conda activate gpu-flopbench && \
    pwd && ls -lah && pip install -r /gpu-flopbench/requirements.txt

# write out to the bashrc to source conda and activate the environment on container startup
RUN echo 'conda activate gpu-flopbench' >> ~/.bashrc

# set an environment variable for convenience
ENV GPU_FLOPBENCH_ROOT=/gpu-flopbench

# expose the Jupyter notebook port
EXPOSE 8888