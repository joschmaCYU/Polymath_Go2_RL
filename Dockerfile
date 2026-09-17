# ==============================================================================
# Dockerfile pour Unitree Go2 Reinforcement Learning (CUDA 12.1 / Driver 535+)
# ==============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# 1. Dépendances système et bibliothèques graphiques OpenGL / EGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1 \
    libgl1 \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libosmesa6-dev \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Créer un lien symbolique pour python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Installation de uv pour des builds ultra-rapides
RUN pip install --no-cache-dir uv

# 3. Installation de PyTorch CUDA 12.1 (compatible avec pilote NVIDIA 535.x)
RUN uv pip install --system --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --system --no-cache scipy tensorboard viser trimesh

WORKDIR /app

# 4. Copie du setup et installation du package
COPY setup.py /app/
COPY src/ /app/src/

RUN uv pip install --system --no-cache --extra-index-url https://download.pytorch.org/whl/cu121 -e .

# 5. Copie du reste des scripts du projet
COPY . /app/

# Port pour le visualiseur Web 3D (Viser)
EXPOSE 8080

CMD ["python", "train_simple.py", "--help"]
