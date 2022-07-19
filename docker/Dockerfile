# This could also be another Ubuntu or Debian based distribution
FROM ubuntu:20.04

# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    libgl1 \
    libgomp1 \
    cmake \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Open3D from the pypi repositories
RUN python3 -m pip install --no-cache-dir --upgrade open3d
RUN pip3 install \
    matplotlib \
    laspy \
    numpy \
    octomap-python \
    glooey \
    pyglet \
    tqdm

# RUN cd /tmp/ && git clone 