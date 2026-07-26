# ============================================================
# Deep Learning Docker Environment
# NVIDIA Blackwell (RTX 50xx) — sm_120
# Requires: CUDA 12.8+, PyTorch 2.6+, Driver 570+
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

# OS dependencies
RUN apt-get update && apt-get install -y \
    wget curl git bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    libgl1-mesa-glx libglu1-mesa \
    build-essential \
    ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda update -n base -c conda-forge conda -y && \
    conda clean -afy

# Copy environment files
WORKDIR /workspace
COPY environment.yml .
COPY requirements.txt .

# Create the conda environment
# CONDA_OVERRIDE_CUDA tells the solver CUDA 12.4 is present even without a GPU at build time
RUN CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yml && \
    conda clean -afy

# Install pip dependencies
RUN /opt/conda/envs/deep-learning/bin/pip install --no-cache-dir \
    -r requirements.txt

# Upgrade PyTorch to cu128 build
RUN /opt/conda/envs/deep-learning/bin/pip install --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Activate deep-learning for all subsequent commands
ENV CONDA_DEFAULT_ENV=deep-learning
ENV PATH=$CONDA_DIR/envs/deep-learning/bin:$PATH

SHELL ["conda", "run", "-n", "deep-learning", "/bin/bash", "-c"]

EXPOSE 8888

# Launch JupyterLab 
CMD ["conda", "run", "--no-capture-output", "-n", "deep-learning", \
     "jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/workspace"]