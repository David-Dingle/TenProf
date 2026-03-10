# TenProf

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7588406.svg)](https://doi.org/10.5281/zenodo.7588406)
[![CodeFactor](https://www.codefactor.io/repository/github/lin-mao/drgpum/badge)](https://www.codefactor.io/repository/github/lin-mao/drgpum)
[![Documentation Status](https://readthedocs.org/projects/drgpum/badge/?version=latest)](https://drgpum.readthedocs.io/en/latest/?badge=latest)


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

## Documentation

- [Installation Guide](https://drgpum.readthedocs.io/en/latest/install.html)
- [User's Guide](https://drgpum.readthedocs.io/en/latest/manual.html)
- [Developer's Guide](https://drgpum.readthedocs.io/en/latest/workflow.html)

## Papers

- Xingjian Ding, Keren Zhou, Yueming Hao, and Pengfei Su. 2026. TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization. The ACM International Conference on Supercomputing, July 6-9, 2026, Belfast, Northern Ireland, UK.