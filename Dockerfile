FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Update packages, install build dependencies, and the specified version of CMake
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget make git gfortran libomp-18-dev libboost-all-dev clang-18 clang-tools-18 && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-linux-x86_64.sh && \
    ./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.0-linux-x86_64.sh


# Set the working directory
WORKDIR /gpu-flopbench

# Copy the requirements file into the container
COPY ./requirements.txt ./requirements.txt

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
    pwd && ls -lah && pip install -r ./requirements.txt 

# Copy the source code into the container
COPY . .

# Config with CMake, and build all the codes
#RUN source ./runBuild.sh 

# Verify that executables were built
#RUN test -n "$(find build -maxdepth 1 -type f -executable)" || (echo "No executables found in ./build" && exit 1)