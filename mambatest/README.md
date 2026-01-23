1. export HF_ENDPOINT=https://hf-mirror.com
2. conda create -n mambatest python=3.10.13 -y
3. conda activate mambatest
4. pip install -r requirements.txt
5. python run.py
## mamba 
location: transformers/src/models/mamba/modeling_mamba.py 
class mambablock
## if want to display modeling_mamba's warning 
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu11torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu11torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl