# MindBNB

quantization for mindnlp

## Setup

apt-get install -y build-essential cmake

pip install -r requirements-dev.txt

cmake -DCOMPUTE_BACKEND=cuda -S .

make
