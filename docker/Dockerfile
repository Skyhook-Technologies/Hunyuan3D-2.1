# docker/Dockerfile

# 1. Base image: CUDA 12.4 + CUDNN + Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04

# 2. Install OS packages (Python3, pip, build tools, OpenGL libs, etc.)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget python3 python3-pip python3-dev \
    build-essential cmake ninja-build \
    libgl1-mesa-glx libegl1-mesa-dev libgles2-mesa-dev \
    libxi6 libxrender1 libsm6 libxext6 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 3. Copy your fork (including inference.py) into the image
COPY .. /workspace

# 4. Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# 5. Install PyTorch 2.5.1 with CUDA 12 support
RUN pip3 install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124/

# 6. Install the rest of Hunyuan3D’s Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# 7. Build & install the custom CUDA extensions
RUN cd hy3dpaint/custom_rasterizer && pip3 install --no-cache-dir -e . && cd /workspace
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd /workspace

# 8. Download the RealESRGAN upscaler weights
RUN mkdir -p hy3dpaint/ckpt \
  && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
         -O hy3dpaint/ckpt/RealESRGAN_x4plus.pth

# 9. (Optional) Patch any relative paths if your fork hasn’t already
RUN sed -i 's|cfgs/hunyuan-paint-pbr.yaml|hy3dpaint/cfgs/hunyuan-paint-pbr.yaml|' hy3dpaint/textureGenPipeline.py
RUN sed -i 's|custom_pipeline = config.custom_pipeline|custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")|' hy3dpaint/utils/multiview_utils.py

# 10. Create & declare mount points for inputs/outputs
RUN mkdir -p /data/in /data/out
VOLUME ["/data/in", "/data/out"]

# 11. Default command: run your batch inference script
CMD ["python3", "inference.py"]
