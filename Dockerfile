FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

WORKDIR /usr/src/app

# Set your timezone
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        git \
        libgl1-mesa-glx \
        build-essential \
        cmake \
        libopencv-dev \
        libopencv-highgui-dev \
        libopencv-imgcodecs-dev \
        libopencv-core-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Clone dlib and build from source with CUDA support
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && mkdir build \
    && cd build \
    && cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 \
    && cmake --build . \
    && cd .. \
    && python3 setup.py install


# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .

# Make sure start.sh is executable
RUN chmod +x start.sh

EXPOSE 8000
ENV PYTHONUNBUFFERED 1

CMD ["./start.sh"]
