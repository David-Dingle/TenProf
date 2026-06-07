# TenProf

TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization

TenProf attributes GPU memory accesses back to the PyTorch **tensors** (and their
Python/C++/CUDA call paths) that produced them. It is built on the newest
[HPCToolkit](https://gitlab.com/hpctoolkit/hpctoolkit), driving NVIDIA
`compute-sanitizer` + `gpu-patch` for memory tracing, [redshow](redshow/) for the
tensor-view analysis, and [torch-monitor](torch-monitor/) for the PyTorch tensor
hooks.

## Quick Start

```bash
gh repo clone git@github.com:David-Dingle/TenProf.git -- --recurse-submodules && cd TenProf

# (Optional) point the installer at a specific PyTorch; otherwise it is
# auto-detected from the active conda env.
export PYTORCH_DIR=path_to_pytorch/torch

# Build & install TenProf into <repo>/tenprof
#   ./bin/install [PREFIX] [CUDA_PATH] [SANITIZER_PATH]
# Build toolchain (meson, cmake>=4.3, patchelf, gawk, ninja) lives in an
# isolated conda env (default: hpctk-build); HPCToolkit's meson subprojects
# auto-fetch dyninst/elfutils/libunwind/xed/xerces-c — no spack needed.
./bin/install

# Profile a PyTorch script (activate your PyTorch env first).
# The tenprof driver self-locates the install and sets PATH/LD_LIBRARY_PATH.
conda activate PYTORCH_ENV_NAME
./bin/tenprof -e torch_view -j THREADS -o out your_script.py [args]
```

## Usage

```bash
tenprof [options] <python-script> [script args]
  -e  <event>     profiling event                  (default: torch_view)
  -j  <threads>   hpcstruct/hpcprof threads         (default: nproc)
  -env <name>     conda env to activate             (default: current env)
  -o  <dir>       output prefix                     (default: tenprof)
  -ck <knob>      extra control knob (repeatable),
                  e.g. -ck HPCRUN_SANITIZER_TORCH_VIEW_ONGPU=1
  -l  <launcher>  launcher prefix, e.g. -l "mpirun -np 1"
  -no-warmup      skip the warmup + hpcstruct passes (analysis only)
  -v              verbose: tee logs to tenprof.log
  -h              help
```

`tenprof` runs four phases:

1. **warmup** — `hpcrun -e gpu=cuda` dumps the GPU cubins.
2. **struct** — `hpcstruct --gpucfg yes` disassembles cubins (+ CPU libraries).
3. **torch_view** — `hpcrun -e gpu=cuda,torch_view` runs `compute-sanitizer` +
   redshow tensor-view analysis.
4. **prof** — `hpcprof` builds the tensor-attributed database.

### Outputs (`out-measurements/` and `out-database/`)

- `out-measurements/torch_view/forest.txt` — the tensor (view) forest.
- `out-measurements/torch_view/torch_view_report.csv` — per-tensor access records.
- `out-measurements/torch_view/torch_view_report.csv.context` — each tensor access
  with its full Python/C++/CUDA call path, produced by hpcprof's TorchView sink
  (correlating redshow's `ctx_id`s with HPCToolkit's calling-context tree).
- `out-database/` — the HPCToolkit `meta.db`/`profile.db`/`cct.db` database.

## Papers

- Xingjian Ding, Keren Zhou, Yueming Hao, and Pengfei Su. 2026. TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization. The ACM International Conference on Supercomputing, July 6-9, 2026, Belfast, Northern Ireland, UK.
