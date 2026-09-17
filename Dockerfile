# ==============================================================================
# Dockerfile pour Unitree Go2 Reinforcement Learning (CUDA 12.1 / Driver 535+)
# ==============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV UV_HTTP_TIMEOUT=300

# 1. Dependances systeme, acceleration materielle GPU et OpenGL / EGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    kmod \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libosmesa6-dev \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Creer un lien symbolique pour python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Installation de uv pour des builds ultra-rapides
RUN pip install --no-cache-dir uv

# 3. Installation de PyTorch CUDA 12.1 (compatible avec pilote NVIDIA 535.x)
RUN uv pip install --system --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Dependances systeme Python et bibliotheques requises pour mjlab et la simulation
RUN uv pip install --system --no-cache \
    scipy tensorboard viser trimesh \
    mujoco==3.5.0 mujoco-warp==3.5.0 warp-lang==1.12.0 \
    tyro tensordict onnx onnxscript wandb mediapy imageio-ffmpeg torchrunx GitPython prettytable tqdm

# 5. Installation de mjlab et rsl-rl-lib sans ecraser PyTorch CUDA
RUN uv pip install --system --no-cache --no-deps mjlab==1.2.0 rsl-rl-lib==5.0.1

WORKDIR /app

# 6. Copie du setup et installation du package local
COPY setup.py /app/
COPY src/ /app/src/

RUN uv pip install --system --no-cache --no-deps -e .

# 5. Copie du reste des scripts du projet
COPY . /app/

# Port pour le visualiseur Web 3D (Viser)
EXPOSE 8080

CMD ["python", "train_simple.py", "--help"]
