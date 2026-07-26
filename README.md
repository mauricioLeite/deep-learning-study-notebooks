# Deep Learning Study Notebooks

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Dockerized JupyterLab environment covering fundamentals through state-of-the-art architectures using PyTorch, TensorFlow, and Hugging Face Transformers.

> **RTX 50-series (Blackwell) users:** the `Dockerfile` and `docker-compose.yml` have already been patched for `sm_120`. See [GPU Requirements](#gpu-requirements) below.

## Table of Contents

- [Project Structure](#project-structure)
- [GPU Requirements](#gpu-requirements)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Notebooks](#notebooks)
- [Datasets](#datasets)
- [Verifying GPU Access](#verifying-gpu-access)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Structure

- `Dockerfile`: CUDA 12.8, Miniforge, and the `deep-learning` environment
- `docker-compose.yml`: service definition with GPU passthrough
- `environment.yml`: pinned Conda environment
- `requirements.txt`: pip packages installed on top of Conda
- `Makefile`: shortcuts for common Docker and JupyterLab operations
- `LICENSE`: MIT license terms
- `notebooks/`: Jupyter notebooks mounted into the container
- `datasets/`: optional datasets mounted into the container

## GPU Requirements

| GPU Generation | Architecture | Compute Capability | Minimum CUDA | Minimum Driver |
|---|---|---|---|---|
| RTX 40-series and older | Ampere / Ada | up to sm_90 | 12.4 | 525+ |
| RTX 50-series (5060 Ti, 5070, etc.) | **Blackwell** | **sm_120** | **12.8** | **570+** |

The image is built from `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` and PyTorch is upgraded to the `cu128` wheel at build time, so it supports both generations out of the box.

## Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Docker Engine | 20.10 | `docker --version` |
| Docker Compose v2 | 2.0 | `docker compose version` |
| NVIDIA driver | 570 (Blackwell) / 525 (others) | `nvidia-smi` |
| nvidia-container-toolkit | any | `nvidia-ctk --version` |

### Install nvidia-container-toolkit (if missing)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

```bash
# 1. Build the image (30 to 50 min on first run; downloads CUDA and Conda packages)
docker compose build

# 2. Start JupyterLab
docker compose up

# 3. Open in a browser
#     http://localhost:8888   (no password required)

# 4. Stop
docker compose down
```

Notebook edits are saved automatically to `./notebooks` on the host. Nothing is lost when the container stops.

## Environment Details

The Conda environment is named **`deep-learning`** (Python 3.12).

### Core frameworks

| Package | Version |
|---|---|
| PyTorch | 2.5.1 base, upgraded to the cu128 build |
| TorchVision | 0.20.1 |
| TensorFlow / Keras | 2.17 / 2.20 / 3.x |
| Hugging Face Transformers | 4.45.2 |
| Hugging Face Datasets | 4.3.0 |
| Accelerate | 1.2.1 |

### Computer vision and detection

| Package | Purpose |
|---|---|
| Ultralytics (YOLOv8/v11) | Object detection and segmentation |
| MONAI | Medical imaging |
| OpenCV | Image processing |
| FAISS-GPU | Similarity search / embeddings |
| scikit-image | Image utilities |
| imgaug | Augmentation |
| DeepFace / MTCNN / RetinaFace | Face analysis |

### NLP and embeddings

| Package | Purpose |
|---|---|
| spaCy 3.7 + `en_core_web_sm/md` | NLP pipeline |
| NLTK | Text utilities |
| Gensim | Word embeddings |
| tiktoken | GPT tokenisation |
| openai-clip | CLIP model |
| torchtext | Text datasets |

### ML utilities

| Package | Purpose |
|---|---|
| scikit-learn | Classical ML |
| UMAP-learn | Dimensionality reduction |
| ONNX / ONNXRuntime-GPU | Model export & inference |
| TensorBoard | Training visualisation |
| Numba | JIT acceleration |

### Notable pip extras (`requirements.txt`)

`pytest`, `coverage`, `pre-commit`, `loguru`, `icecream`, `torch-summary`, `torchsummary`, `selectivesearch`, `minisom`, `einops`, `yt-dlp`, `gdown`, `flask`, `pydantic`

## Notebooks

Notebooks live in `./notebooks/` and are served directly by JupyterLab.

### Foundations

| Notebook | Topic |
|---|---|
| `1-introducing-pytorch.ipynb` | Tensors, autograd, training loop basics |
| `2-examples-of-dataset-types.ipynb` | `Dataset` / `DataLoader` patterns |
| `3-simple-classification.ipynb` | Binary and multi-class classification |
| `4-simple-regression.ipynb` | Linear regression with PyTorch |
| `5-activation-functions.ipynb` | ReLU, Sigmoid, Tanh, GELU, etc. |
| `6-dataset-preparation.ipynb` | Transforms, splits, custom datasets |

### Convolutional Neural Networks

| Notebook | Topic |
|---|---|
| `7-first-image-neuralnet.ipynb` | First image classifier, from MLP to CNN |
| `8-introducing-pytorch-convnet.ipynb` | ConvNet building blocks |
| `9-pytorch-convnet.ipynb` | Full ConvNet training pipeline |
| `10-pytorch-using-pretrained-convnet.ipynb` | Transfer learning with pretrained models |
| `14-TransposeConvolution2D.ipynb` | Deconvolution and upsampling layers |

### Object Detection and Segmentation

| Notebook | Topic |
|---|---|
| `12-candidate_objects.ipynb` | Selective search for region proposals |
| `13-ft_yolo_test.ipynb` | Fine-tuning YOLOv8 for custom detection |
| `15-Semantic_Segmentation_with_U_Net.ipynb` | U-Net architecture for pixel-level segmentation |

### Representation Learning

| Notebook | Topic |
|---|---|
| `11-contrastive-learning.ipynb` | SimCLR-style self-supervised learning |

### Transformers and NLP

| Notebook | Topic |
|---|---|
| `16-text-classification-sentiment-analysis.ipynb` | Fine-tuning BERT for sentiment classification |
| `17-GPTFromScratch.ipynb` | Implementing a GPT decoder from scratch |
| `18-pytorch-ViT.ipynb` | Vision Transformer (ViT) for image classification |
| `19-pytorch-SegFormer.ipynb` | SegFormer for semantic segmentation |

## Datasets

Place large dataset files under `./datasets/`. The directory is volume-mounted at `/workspace/datasets` inside the container.

Notebooks that need to download datasets (e.g. via `gdown`, `yt-dlp`, or Hugging Face `datasets`) will do so automatically on first run.

## Verifying GPU Access

From a JupyterLab terminal or a new notebook cell:

```python
import torch
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())
print("GPU     :", torch.cuda.get_device_name(0))
print("Compute :", torch.cuda.get_device_capability())  # (12, 0) for Blackwell
```

Or from the host:

```bash
docker exec -it deep-learning_jupyter conda run -n deep-learning python -c "
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA capability sm_120 is not compatible` | The cu124 PyTorch wheel is being used. Force a clean rebuild: `docker compose build --no-cache` |
| `nvidia-smi` works on host but not inside container | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `faiss-gpu` install failure | Replace `faiss-gpu` with `faiss-cpu` in `environment.yml`. The notebooks continue to work. |
| TensorFlow notebooks ignore the GPU | TF may not ship sm_120 kernels yet. Add `import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''` at the top of the affected notebook to fall back to CPU while keeping PyTorch on GPU |
| `Bus error` / DataLoader worker crash | `shm_size` is already set to 16 GB in `docker-compose.yml`. If crashes persist, reduce `num_workers` in the DataLoader |
| Port 8888 already in use | Change the host port in `docker-compose.yml`: `"8889:8888"` |

## License

This project is licensed under the [MIT License](LICENSE).
