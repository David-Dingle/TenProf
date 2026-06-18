# TenProf

TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization

TenProf attributes GPU memory **stalls** back to the PyTorch **tensors** (and their
Python/C++/CUDA call paths) that produced them. It is built on the newest
[HPCToolkit](https://gitlab.com/hpctoolkit/hpctoolkit), and merges two analyses in a
single profiling run:

- **tensor attribution** — NVIDIA `compute-sanitizer` + `gpu-patch` trace every GPU
  memory access; [redshow](redshow/)'s torch-view builds the tensor (view) forest;
  [torch-monitor](torch-monitor/) supplies the PyTorch tensor hooks.
- **memory-stall blame-shift** — the CUPTI **PC Sampling** API collects per-PC memory
  stalls, which are blame-shifted to the producing load instruction.

`kernel_replay` mode joins the two by `(kernel_id, pc)` so each memory stall is charged
to the tensor it was waiting on, producing the **Tensor Stall Cost** and the tensor
transformation forest (paper Figure 7).

## Build

```bash
gh repo clone git@github.com:David-Dingle/TenProf.git -- --recurse-submodules && cd TenProf

# (Optional) point the installer at a specific PyTorch; otherwise it is
# auto-detected from the active conda env.
export PYTORCH_DIR=path_to_pytorch/torch

# Build & install TenProf into <repo>/tenprof
#   ./bin/install [PREFIX] [CUDA_PATH] [SANITIZER_PATH]
./bin/install
```

`bin/install` builds the whole stack into `<repo>/tenprof/` in dependency order:

1. **gpu-patch** — compute-sanitizer fatbins (`make install`; `ARCHS` includes sm_89).
2. **HPCToolkit meson *setup*** — configures the build and fetches the meson
   subprojects (dyninst, elfutils, libunwind, xed, xerces-c). No spack required.
3. **torch-monitor** — CMake, built against the **active env's** PyTorch
   (`RelWithDebInfo`, `-O3`).
4. **redshow** — the torch-view analysis library (`-O3`), linked against gpu-patch +
   torch-monitor + the libunwind headers fetched in step 2.
5. **HPCToolkit meson *compile + install*** — links `libredshow` + `libsanitizer-public`
   into `libhpcrun.so`, then `patchelf`s `$ORIGIN` rpaths so the install is relocatable.

The build toolchain (meson, cmake≥4.3, patchelf, gawk, ninja) lives in an isolated
conda env (default `hpctk-build`, override with `TENPROF_BUILD_ENV`); your PyTorch env
stays untouched. CUDA defaults to `/usr/local/cuda` — pass a different `CUDA_PATH` as
the 2nd argument if needed.

## Profile (`kernel_replay`)

```bash
conda activate <pytorch-env>          # the driver self-locates the install + sets paths
./bin/tenprof -o out your_script.py [script args]      # -e kernel_replay is the default
```

`kernel_replay` runs as **five phases** (the driver sets `PATH`/`LD_LIBRARY_PATH` and
preloads `libgcc_s` so torch C++ exceptions unwind correctly under hpcrun):

| # | phase | command | produces |
|---|-------|---------|----------|
| 1 | **warmup**  | `hpcrun -e gpu=cuda`                | dumps GPU cubins (no sanitizer, fast) |
| 2 | **struct**  | `hpcstruct --gpucfg yes`            | disassembles cubins (+ CPU libs) → `structs/` |
| 3 | **profile** | `hpcrun -e gpu=cuda,kernel_replay`  | **one** run: per-kernel checkpoint replay does the sanitizer (tensor accesses) **and** PC sampling (memory stalls) in a single process |
| 4 | **blame**   | `tensor_blame <gpubins> <meas>`     | blame-shifts stalls to producing loads, joins on `(kernel_id, pc)` → `tensor_stall_cost.csv` + `blame.dot`; `dot -Tsvg` → `blame.dot.svg` |
| 5 | **prof**    | `hpcprof -o <db> <meas>`            | the HPCToolkit `meta.db`/`profile.db`/`cct.db` database |

Why a single run? The compute-sanitizer and the CUPTI PC-sampling profiler are
**mutually exclusive** CUPTI clients and cannot share a process simultaneously.
`kernel_replay` checkpoints and replays each kernel, running the sanitizer pass and the
PC-sampling pass per kernel within one process, so they share a valid per-kernel
`correlationId` and join cleanly by `kernel_id` — no second run, no `pystates_hash`
bridge needed.

## Usage

```bash
tenprof [options] <python-script> [script args]
  -e  <event>     kernel_replay (default) | torch_view | pc_sampling | both
  -j  <threads>   hpcstruct/hpcprof threads          (default: nproc)
  -env <name>     conda env to activate              (default: current env)
  -o  <dir>       output prefix                      (default: tenprof)
  -ck <knob>      extra control knob (repeatable),
                  e.g. -ck HPCRUN_SANITIZER_TORCH_VIEW_ONGPU=1
  -l  <launcher>  launcher prefix, e.g. -l "mpirun -np 1"
  -no-warmup      skip the warmup + hpcstruct passes (use cached data)
  -no-prof        skip the hpcprof phase (inst stalls -> tensors)
  -v              verbose: tee logs to tenprof.log
  -h              help
```

Events:

- **`kernel_replay`** *(default)* — single-run merge: tensor accesses **+** memory
  stalls, joined per kernel. Produces `tensor_stall_cost.csv` + `blame.dot`.
- **`torch_view`** — tensor accesses only (`compute-sanitizer` + redshow).
- **`pc_sampling`** — memory stalls only (CUPTI PC Sampling).
- **`both`** — *application replay*: a whole-application `pc_sampling` run then a
  whole-application `torch_view` run into the same measurements dir. Because the two
  runs are separate processes, they are joined on the content-based `(pystates_hash,
  pc)` key (not `kernel_id`, which is process-local). `tensor_blame` then runs and
  produces the same `tensor_stall_cost.csv` + `blame.dot` as `kernel_replay`. Use this
  when per-kernel `kernel_replay` is too slow or unstable on a given model.

## Outputs (`out-measurements/`)

- `out-measurements/tensor_stall_cost.csv` — **the result**: memory stall cost charged
  to each tensor (kernel_replay).
- `out-measurements/blame.dot` (+ `.svg`) — the tensor transformation forest (paper
  Figure 7): base = solid ellipse, view = dashed ellipse, node size ∝ stall %, the
  blamed path in red. Rendered by plain graphviz `dot`.
- `out-measurements/pc_sampling/pc_samples.csv` — per-PC memory stalls (`kernel_id` + `pc`).
- `out-measurements/torch_view/forest.txt` — the tensor (view) forest.

## Papers

The full paper is included in this repository: [reference/TenProf.pdf](reference/TenProf.pdf).

- Xingjian Ding, Keren Zhou, Yueming Hao, and Pengfei Su. 2026. TenProf: A Tensor-Centric Profiler for Deep Learning Workload Analysis and Optimization. The ACM International Conference on Supercomputing, July 6-9, 2026, Belfast, Northern Ireland, UK. [[PDF](reference/TenProf.pdf)]
