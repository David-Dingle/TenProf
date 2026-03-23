# TenProf

TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization
## Quick Start

```bash
gh repo clone git@github.com:David-Dingle/TenProf.git -- --recurse-submodules && cd TenProf

# Specify PyTorch dir
export PYTORCH_DIR=path_to_pytorch/torch

# Install DrGPUM
./bin/install

# Setup environment variables
export TenProf_PATH=$(pwd)/gvprof
export PATH=${TenProf_PATH}/bin:$PATH
export PATH=${TenProf_PATH}/hpctoolkit/bin:$PATH
export PATH=${TenProf_PATH}/redshow/bin:$PATH

# Test a sample
conda activate PYTORCH_ENV_NAME
gvprof -env PYTORCH_ENV_NAME -v -cfg -j THREADS -e torch_view pytorch_exec.py args
```


## Papers

- Xingjian Ding, Keren Zhou, Yueming Hao, and Pengfei Su. 2026. TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization. The ACM International Conference on Supercomputing, July 6-9, 2026, Belfast, Northern Ireland, UK.