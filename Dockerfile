FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

WORKDIR /usr/src/app

# Set timezone
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        gcc-9 g++-9 \
        git \
        build-essential \
        cmake \
        libgtk2.0-dev \
        pkg-config \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        python3-dev \
        python3-numpy \
        nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Install cuDNN
COPY cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb /tmp
RUN dpkg -i /tmp/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb 
RUN cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-8AE81B24-keyring.gpg /usr/share/keyrings/ 
RUN apt-get update \
    && apt-get install -y libcudnn8=8.9.7.29-1+cuda11.8 libcudnn8-dev=8.9.7.29-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/* \
    && rm /tmp/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb

# Clone and build OpenCV from source
RUN git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
             -D CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ \
             -D CUDA_HOST_COMPILER=/usr/bin/gcc-9 \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D INSTALL_C_EXAMPLES=OFF \
             -D INSTALL_PYTHON_EXAMPLES=ON \
             -D WITH_CUDA=ON \
             -D CUDA_ARCH_BIN=8.7 \
             -D CUDA_ARCH_PTX="" \
             -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
             -D BUILD_EXAMPLES=OFF .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd ../../ \
    && rm -rf opencv opencv_contrib

# Clone and build dlib from source with CUDA support
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && mkdir build \
    && cd build \
    && cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DCUDA_HOST_COMPILER=/usr/bin/gcc-9 -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ \
    && cmake --build . \
    && cd .. \
    && python3 setup.py install \
    && cd .. \
    && rm -rf dlib

RUN pip install face-recognition
COPY . .

RUN chmod +x start.sh

EXPOSE 8000
ENV PYTHONUNBUFFERED 1

CMD ["./start.sh"]
