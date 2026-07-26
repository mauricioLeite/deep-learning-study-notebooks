# ============================================================
# Deep Learning Environment
# ============================================================

SERVICE = deep-learning
CONTAINER = deep-learning_jupyter
JUPYTER = http://localhost:8888

.PHONY: help build up down restart logs shell clean rebuild status open

help:
	@echo ""
	@echo "  Deep Learning Docker Environment"
	@echo ""
	@echo "  make build      Build the Docker image"
	@echo "  make up         Start JupyterLab (detached)"
	@echo "  make down       Stop the container"
	@echo "  make restart    Restart the container"
	@echo "  make logs       Follow container logs"
	@echo "  make shell      Open bash inside the container"
	@echo "  make conda      Open bash with conda env active"
	@echo "  make gpu        Show GPU status (nvidia-smi)"
	@echo "  make status     Show running containers"
	@echo "  make open       Print JupyterLab URL"
	@echo "  make clean      Remove container + volumes"
	@echo "  make rebuild    Full rebuild from scratch (no cache)"
	@echo ""

# Core

build:
	docker compose build

up:
	docker compose up -d
	@echo "JupyterLab running at $(JUPYTER)"

down:
	docker compose down

restart:
	docker compose restart $(SERVICE)

logs:
	docker compose logs -f $(SERVICE)

status:
	docker compose ps

# Shell
shell:
	docker exec -it $(CONTAINER) bash

conda:
	docker exec -it $(CONTAINER) conda run --no-capture-output -n deep-learning bash

# GPU
gpu:
	docker exec -it $(CONTAINER) nvidia-smi

cuda-check:
	docker exec -it $(CONTAINER) conda run -n deep-learning python -c \
		"import torch; print('PyTorch:', torch.__version__); \
		 print('CUDA available:', torch.cuda.is_available()); \
		 print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Maintenance 

open:
	@echo "JupyterLab → $(JUPYTER)"

clean:
	docker compose down -v --remove-orphans

rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	@echo "Rebuilt and running at $(JUPYTER)"

purge: clean
	docker rmi deep-learning:latest || true