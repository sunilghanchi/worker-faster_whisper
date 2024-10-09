# Use specific version of nvidia cuda image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV HF_HOME=/root/.cache/huggingface

# Set working directory
WORKDIR /

# Add deadsnakes PPA and install Python 3.10
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install --yes --no-install-recommends python3.10 python3.10-venv python3.10-dev libgl1 libx11-6 ffmpeg build-essential && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Remove the existing python3 and link python3.10
RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Download and install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt

# Copy and run script to fetch models
COPY builder/fetch_models.py /fetch_models.py

# Add debugging information
RUN echo "Running fetch_models.py" && \
    python /fetch_models.py && \
    echo "fetch_models.py completed" && \
    rm /fetch_models.py

# Copy source code into image
COPY src .

# Set default command
CMD ["python", "-u", "/rp_handler.py"]
