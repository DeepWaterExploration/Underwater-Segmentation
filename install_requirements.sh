#!/bin/bash

source .venv/bin/activate 

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install onnx-runtime onnx
pip install onnxruntime
pip install opencv-python
pip install Pillow
pip install matplotlib
pip install scipy