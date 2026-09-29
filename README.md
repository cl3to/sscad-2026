# Comparing MPI, MPP, and StarPU Runtimes for Distributed GPU Execution: A Case Study Using N-Body Simulations

This repository contains the artifact for the paper accepted at **SSCAD 2026** (XXVII Simpósio em Sistemas Computacionais de Alto Desempenho).

The work presents a controlled comparison of three approaches to distributed GPU execution on a cluster of AMD Instinct MI210 GPUs:

- **MPI** — the conventional SPMD message-passing approach, combined with either OpenMP `target` offloading or native HIP kernels.
- **MPP (MPI Proxy Plugin)** — an extension of the LLVM OpenMP `liboffload` runtime (built on the OpenMP Cluster Model, OMPC) that intercepts `#pragma omp target` regions issued by an unmodified OpenMP application and forwards them, via asynchronous MPI and C++20 coroutines, to accelerators attached to *remote* nodes. Each remote GPU is served by a proxy process (`llvm-offload-mpi-proxy-device`), so the cluster appears to the host process as a single pool of OpenMP target devices.
- **StarPU** — a task-based runtime (full-MPI mode) where the application registers data handles and submits codelets; the runtime infers dependencies from read/write accesses and issues inter-node transfers automatically under an owner-computes policy.

Two benchmarks are used: a **direct-method N-Body** simulation (O(n²), fixed problem size of 2²⁰ bodies) for strong-scaling analysis, and **Task Bench** (Slaughter et al., SC20) for characterizing the impact of task dependency patterns (`trivial`, `tree`, `stencil_1d`) through the METG(50%) metric. A single-GPU **roofline analysis** of the `bodyForce` and `integratePositions` kernels separates kernel-implementation effects (HIP vs. OpenMP) from runtime effects.

## Repository Structure

```
.
├── launch.sh                    # Submit a sweep of Slurm jobs (1 to N workers)
│
├── yml/                         # Spinner YAML experiment definitions
│   ├── nbody.yml                # N-Body strong scaling (MPI/MPP/StarPU × OpenMP/HIP)
│   ├── taskbench.yml            # Task Bench, CPU variants (exploratory)
│   ├── taskbench_gpu.yml        # Task Bench GPU, strong scaling
│   └── taskbench_gpu_ws.yml     # Task Bench GPU, weak scaling (METG experiments)
│
├── *.job                        # Slurm job submission templates
│   ├── nbody.job
│   ├── taskbench.job
│   ├── taskbench_gpu.job        # used for both strong and weak scaling runs
│   └── profile.job              # rocprofv3 profiling for the roofline figure
│
├── figures/                     # Final figures reproduced by the artifact (PDF)
│
└── graphs/                      # Analysis pipeline
    ├── sscad-26.ipynb           # Jupyter notebook reproducing all figures
    ├── utils.py                 # Data loading, aggregation, plotting, METG, roofline
    ├── out/                     # Experimental results (Spinner pickle files)
    ├── figures/                 # Intermediate and final figures (PDF)
    ├── hip_kernel_stats.csv     # rocprofv3 kernel stats (HIP variant)
    ├── hip_kernel_trace.csv     # rocprofv3 kernel dispatch trace (HIP variant)
    ├── openmp_kernel_stats.csv  # rocprofv3 kernel stats (OpenMP target variant)
    └── openmp_kernel_trace.csv  # rocprofv3 kernel dispatch trace (OpenMP target variant)
```

`logs/` contains the Slurm stdout of the reported runs.

## The Runtimes Compared

| Runtime      | Execution model | GPU kernel implementation                              |
|--------------|-----------------|--------------------------------------------------------|
| MPI          | SPMD ranks with explicit `MPI_Bcast`/`Scatterv`/`Allgather` | OpenMP `target` regions (`bin/nbody-mpi-openmp`) or native HIP (`bin/nbody-mpi-hip`) |
| MPP          | Single OpenMP process + one proxy process per remote GPU (MPMD) | OpenMP `target` only (`bin/nbody-ompc`)               |
| StarPU       | Full-MPI mode, task graph over data handles, owner-computes transfers | Codelets with OpenMP `target` (`bin/nbody-starpu-openmp`) or HIP (`bin/nbody-starpu-hip`) |

The MPP execution follows a two-level layout visible in the Spinner command templates:

```
mpirun -np <workers> ... llvm-offload-mpi-proxy-device :   # proxies, one GPU set each
       -np 1 ... bin/nbody-ompc 20                          # the (unmodified) OpenMP host
```

The host offloads `target` regions to the proxy devices; each proxy forwards the kernel and its `map`ped data to the accelerator(s) it owns over MPI, hiding the network entirely behind the `liboffload` plugin interface.

## Hardware and Software Stack

All experiments were conducted on the **AMD University Program (AUP) AI and HPC cluster**:

| Item          | Specification                                              |
|---------------|------------------------------------------------------------|
| Nodes         | 1–8 compute nodes (partition `mi2104x`), exclusive allocation |
| CPUs          | 2× AMD EPYC 7V13 per node (128 cores total per node)       |
| GPUs          | 4× AMD Instinct MI210 per node, AMD Infinity Fabric        |
| Network       | NVIDIA ConnectX-6 MT28908 InfiniBand (200 Gb/s per adapter) |
| Scheduler     | Slurm                                                     |
| Compiler      | Clang 21.0 with OMPC/MPP-enabled LLVM OpenMP runtime       |
| GPU stack     | ROCm 7.2.0                                                |
| MPI           | MPICH 4.3.2 + UCX 1.20 (GPU-aware communication enabled)  |
| Task runtime  | StarPU 1.4.12                                             |
| Experiment mgmt | Spinner (YAML-driven parameter sweeps)                  |
| Virtualization | Apptainer containers (`--rocm --sharens`)                 |

Each experiment was repeated **five times** (fewer for some Task Bench sweeps); reported results are the **mean execution time with 95% confidence intervals** estimated via bootstrap resampling (drawn as semi-transparent error bands).

## Dependencies

### Building the Benchmark Applications

The binaries are *not* included in this repository. They come from the two upstream projects below and must be installed under `bin/` inside the Apptainer images (see paths in `yml/*.sif` references).

**1. N-Body** — [github.com/cl3to/N-Body-OMPC](https://github.com/cl3to/N-Body-OMPC), branch `ompc`:

```bash
git clone -b ompc https://github.com/cl3to/N-Body-OMPC.git
# Each variant has its own Makefile under src/<variant>:
make -C src/ompc        # -> bin/nbody-ompc          (MPP, pure OpenMP)
make -C src/mpi         # -> bin/nbody-mpi-openmp
make -C src/mpi-hip     # -> bin/nbody-mpi-hip
make -C src/starpu-omp  # -> bin/nbody-starpu-openmp
make -C src/starpu-hip  # -> bin/nbody-starpu-hip
```

Compile with a Clang/LLVM built with the MPP runtime (`OMPTARGET_*` plugin support); HIP variants additionally require HIP/ROCm. See `Dockerfile.openmp` and `Dockerfile.starpu` upstream for reference environments.

**2. Task Bench** — [gitlab.com/ompcluster/task-bench](https://gitlab.com/ompcluster/task-bench/-/tree/ompc-multidevice), branch `ompc-multidevice` (contains the `mpp`, `mpi_openmp` and `starpu` ports of the benchmark with a custom compute-bound GPU kernel):

```bash
git clone -b ompc-multidevice https://gitlab.com/ompcluster/task-bench.git
cd task-bench
./get_deps.sh
./build_all.sh        # builds core + the runtime ports enabled via env flags
```

Expected binaries (CPU and GPU variants, named as referenced by the YAMLs):

- `bin/tb_mpi_openmp`, `bin/tb_starpu`, `bin/tb_mpp`
- `bin/tb_gpu_mpi_openmp`, `bin/tb_gpu_starpu`, `bin/tb_gpu_mpp`

### Containers

The Spinner configs reference two Apptainer images:

- `mpp.sif` — OMPC/MPP-enabled LLVM + MPICH runtime, ships `llvm-offload-mpi-proxy-device` and the MPP benchmark binaries.
- `rocm-openmp-mpich-starpu_latest.sif` — ROCm + OpenMP + MPICH + StarPU stack for the MPI and StarPU variants.

Both expose the benchmarks at `bin/` relative to the working directory.

### Python (Analysis)

`spinner` (inside a virtualenv, e.g. `$WORK/sscad-2025/.venv`), plus `pandas`, `numpy`, `seaborn`, `matplotlib` for the notebook.

## Reproducing the Experiments

### Running via Slurm

`launch.sh` sweeps the number of workers (nodes): it copies the base YAML, rewrites the per-application `workers` field into `yml/run_<i>.yml`, and submits one job per worker count.

```bash
# N-Body strong scaling, 1–8 workers (4–32 GPUs), Figure 4
YAML=yml/nbody.yml JOB_SCRIPT=nbody.job NS=1 NE=8 bash launch.sh

# Task Bench GPU strong scaling, 1–10 workers
YAML=yml/taskbench_gpu.yml JOB_SCRIPT=taskbench_gpu.job NS=1 NE=10 bash launch.sh

# Task Bench GPU weak scaling (METG curves, Figures 2 and 3)
YAML=yml/taskbench_gpu_ws.yml JOB_SCRIPT=taskbench_gpu.job NS=1 NE=10 bash launch.sh

# Task Bench CPU sweep (exploratory)
YAML=yml/taskbench.yml JOB_SCRIPT=taskbench.job NS=1 NE=4 bash launch.sh
```

Each `.job` script resolves the allocated node list into `$hosts`, activates the Spinner virtualenv, and writes results to `out/<date>_<bench>_<nodes>_workers.pkl` (dated by the `rel` variable, e.g. `080826`).

### Experiment Parameters

| Experiment               | YAML                   | Workers | GPUs/Worker | Sweeps                                                        |
|--------------------------|------------------------|---------|-------------|---------------------------------------------------------------|
| N-Body strong scaling    | `nbody.yml`            | 1–8     | 4           | 5 runtimes × sizes (problem fixed at `20` ≈ 2²⁰ bodies)       |
| Task Bench GPU (SS)      | `taskbench_gpu.yml`    | 1–8     | 4           | `{stencil_1d, trivial, tree}` × `-iter` 2²…2²⁰               |
| Task Bench GPU (WS/METG) | `taskbench_gpu_ws.yml` | 1–10    | 1 (SS runs also used) | 3 patterns × `-iter` 2²…2²⁰, width scaled with workers |
| Task Bench CPU           | `taskbench.yml`        | 1–4     | —           | 5 graph types × `10¹…10⁸` iterations (MPI+OpenMP, StarPU)     |

### Profiling for the Roofline (Figure 5)

```bash
sbatch profile.job
```

This runs the N-Body host binary (2²⁰ bodies) under `rocprofv3` (`--stats --sys-trace --output-format pftrace`) on 4 nodes, separately for the HIP and OpenMP `target` kernel versions. The resulting per-kernel statistics and dispatch traces are the CSVs under `graphs/` (`hip_kernel_{stats,trace}.csv`, `openmp_kernel_{stats,trace}.csv`). Kernel performance is computed as total floating-point operations divided by the **sum of kernel dispatch durations** (excluding gaps between iterations).

### Running Manually (without Slurm)

Each job reduces to a Spinner invocation; e.g.:

```bash
spinner -vv run yml/nbody.yml \
  -e hosts=node01,node02,... \
  -o out/080826_nbody_8_workers.pkl
```

Key environment variables embedded in the YAML command templates:

- `OMPTARGET_NUM_EXEC_EVENT_HANDLERS=4` / `OMPTARGET_NUM_DATA_EVENT_HANDLERS=4` — MPP proxy handler threads
- `OMPTARGET_MPI_FRAGMENT_SIZE=2000000000` — MPI transfer fragmentation buffer
- `OMPTARGET_NUM_MPI_COMMS` / `MPIR_CVAR_CH4_NUM_VCIS` — MPI communication channels / VCs
- `MPIR_PARAM_CH4_GLOBAL_PROGRESS=0` — disables MPICH global progress polling
- `ROCR_VISIBLE_DEVICES` — GPU visibility mask per rank (`dlist`)
- `OMPTARGET_DISABLE_HOST_PLUGIN=1` — host rank sees no local devices (MPP runs)
- `KMP_AFFINITY=granularity=thread,scatter` + `numactl --cpunodebind/--membind` — NUMA pinning of proxies vs. host

## Generating the Figures

Open and execute `graphs/sscad-26.ipynb` (run it from inside `graphs/`, it loads the pickles from `out/` and writes PDFs to `figures/`):

| Notebook section                | Output figure                                | Paper reference |
|--------------------------------|----------------------------------------------|-----------------|
| N-Body                         | `{DATE}_nbody_strong_runtime.pdf`            | Figure 4        |
| Task Bench — efficiency curves | `140826_ws_trivial_metg.pdf`                 | Figure 2        |
| Task Bench — relative METG     | `130826_taskbench_ws_metg_relative.pdf`      | Figure 3        |
| Task Bench — runtime sweeps    | `{DATE}_taskbench_runtime.pdf`, `{DATE}_taskbench_ws.pdf`, `{DATE}_ws_metg.pdf` | supporting data |
| N-Body roofline                | `130826_nbody_roofline_1gpu.pdf`             | Figure 5        |

`utils.py` provides:

- `load_data()` / `aggregate_data()` — Spinner pickle loading and mean/std aggregation
- `generic_line_plot_err_bar*()` — Seaborn line plots with 95% bootstrap CI bands
- `find_metg()` — computes **METG(50%)**: task granularity `G = (T × N_GPU × 10³) / N_tasks` (ms per task per GPU), with efficiency normalized to the maximum throughput of the same runtime/GPU-count/output/pattern; METG is the granularity where the curve crosses 50% efficiency (linear interpolation). Lower = finer tasks sustained.
- `metg_relative_bar_subplots_2()` — relative METG, `METG(p) / METG(trivial)`, per runtime and GPU count
- `nbody_roofline()` — roofline of the MI210 (22.6 TFLOP/s FP32 peak, 1.6 TB/s HBM2e bandwidth, ridge point ≈ 14.1 FLOP/byte) with the traffic model from the paper: `F_bodyForce = 18·N_target·N_bodies + 6·N_target`, `F_integrate = 6·N_target`, `B_bodyForce = 12·N_bodies + 36·N_target`, `B_integrate = 36·N_target` (12-byte structs, source positions reused once per dispatch).

## Logs and Results

- **Logs**: Slurm stdout captured to `logs/{benchmark}_{jobid}.out`
- **Results**: Spinner pickle files in `graphs/out/` (`dataframe` + `metadata`), named `{DDMMYY}_{bench}_{workers}_workers.pkl`
- **Figures**: high-resolution PDFs in `figures/` (final) and `graphs/figures/` (analysis output)

## Citation

If you use this work, please cite the SSCAD 2026 paper:

```
@inproceedings{cleto_sscad2026,
  title     = {Comparing {MPI}, {MPP}, and {StarPU} Runtimes for Distributed {GPU}
               Execution: A Case Study Using {N-Body} Simulations},
  author    = {Jhonatan Cl{\'e}to and Nicolas Vanz and Herv{\'e} Yviquel and
               M{\'a}rcio Castro},
  booktitle = {Anais do XXVII Simp{\'o}sio em Sistemas Computacionais de Alto
               Desempenho (SSCAD)},
  year      = {2026}
}
```

Related upstream work:

- Cléto, Valarini, Pereira, Araujo, Yviquel. *Scalable OpenMP Remote Offloading via Asynchronous MPI and Coroutine-Driven Communication* (MPP). Euro-Par 2025.
- Yviquel et al. *The OpenMP Cluster Programming Model* (OMPC). ICPP-W 2023.
- Slaughter et al. *Task Bench: A Parameterized Benchmark for Evaluating Parallel Runtime Performance*. SC20.
- Vanz, Munhoz, Castro. *N-Body Problem with StarPU/OpenMP* (base application). SSCAD 2025.
