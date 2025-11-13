FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ARG HOSTOS=linux 

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Update packages, install build dependencies, and the specified version of CMake
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget make git gfortran libomp-18-dev libboost-all-dev clang-18 clang-tools-18 unzip && \
    apt-get install -y imagemagick && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-linux-x86_64.sh && \
    ./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.0-linux-x86_64.sh

# setup alternative names for the compiler
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-18 100


# Set the working directory
WORKDIR /gpu-flopbench

# Copy the requirements file into the container
COPY ./requirements.txt ./requirements.txt

# if we are on a windows host, we need to remove the CRLF characters from all the files
RUN sed -i 's/\r$//' requirements.txt

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

# if we are on a windows host, we need to remove the CRLF characters from all the files
#RUN find . -type f -exec sed -i 's/\r$//' {} +
RUN if [ "$HOSTOS" = "windows" ]; then \
    echo "Removing CRLF characters from files..."; \
    find . -type f \
        -not -name "*.gz" \
        -not -name "*.zip" \
        -not -name "*.rar" \
        -not -name "*.7z" \
        -not -name "*.tar" \
        -not -name "*.bz2" \
        -not -name "*.tar.*" \
        -exec sed -i 's/\r$//' {} +; \
    else \
    echo "HOSTOS is not windows; skipping CRLF removal."; \
    fi 

# write out to the bashrc to source conda and activate the environment on container startup
RUN echo 'conda activate gpu-flopbench' >> ~/.bashrc

# one of the issues with a windows host is that the execute permissions are not preserved when copying files into the container
# this is okay, and it seems like everything works fine without needing to change it.