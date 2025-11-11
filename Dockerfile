FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update packages, install build dependencies, and the specified version of CMake
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget make git gfortran libomp-18-dev libboost-all-dev clang-18 clang-tools-18 && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-linux-x86_64.sh && \
    ./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.0-linux-x86_64.sh

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    source ~/anaconda3/bin/activate && \
    conda init --all && \
    conda create --name "hecbench-roofline" python=3.11.11 && \
    conda activate hecbench-roofline && \
    pip install -r ./requirements.txt 


# Set the working directory
WORKDIR /hecbench-roofline

# Copy the source code into the container
COPY . .

# Config with CMake, and build all the codes
RUN source ./runBuild.sh 

# Verify that executables were built
RUN test -n "$(find build -maxdepth 1 -type f -executable)" || (echo "No executables found in ./build" && exit 1)