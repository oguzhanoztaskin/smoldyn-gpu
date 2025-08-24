FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Needed for OpenGL
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Install system dependencies including development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    pkg-config \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglew-dev \
    # libzlib1g-dev \
    libpng-dev \
    libboost-all-dev \
    python3 \
    python3-pip \
    gdb \
    valgrind \
    vim \
    nano \
    htop \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install CMake 3.18
RUN wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    ln -sf /usr/local/bin/cmake /usr/bin/cmake && \
    ln -sf /usr/bin/make /usr/bin/gmake

# Set working directory
WORKDIR /workspace

# Copy source code for development
COPY smoldyn-gpu-gladkov/ /workspace/smoldyn-gpu-gladkov/

# Set up development environment
RUN echo 'alias ll="ls -la"' >> /root/.bashrc && \
    echo 'alias ..="cd .."' >> /root/.bashrc && \
    echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc

# Default command for development
CMD ["/bin/bash"]

# Add labels for documentation
LABEL maintainer="Smoldyn GPU Docker Image"
LABEL description="Docker container for developing Smoldyn GPU molecular dynamics simulation (Development)"
LABEL version="1.0-dev"
