# Smoldyn GPU Docker Setup

This repository provides Docker-based development and deployment for Smoldyn GPU molecular dynamics simulation based on [Denis Gladkov's implementation](https://www.smoldyn.org/download.html).

Fixed a few compile errors, bumped CUDA architecture from 20 to 89 (mine), removed now obsolete CUDPP with thrust library (which was already present in the project). CUDPP was only used for nvRadixSort, I replaced it with thrust's sort which is not a radix sort but a comparison based sort according to [here:](https://nvidia.github.io/cccl/thrust/api/group__sorting_1ga857dfc9b50a03888c5703699f0fbfcd4.html) `This version of sort compares objects using operator<.`
Therefore, despite in the code this sort is called "radixsort", it is not. I am yet to refactor it to be called just sort.

> [!WARNING]
> The [paper](https://www.smoldyn.org/andrews/papers/Gladkov_Andrews_2011.pdf) uses radixsort which is implemented in the original code but currently we are using the comparison based sort, `thrust::sort`. In the code, the sort is misnamed.

## Architecture

Two Docker images are provided:

1. **Development Image** (`smoldyn-gpu-dev`): Contains all development tools including GDB, Valgrind, and editors. Source code is mounted as a volume for live development.
2. **Release Image** (`smoldyn-gpu-release`): Optimized runtime image with Smoldyn built in release mode. Minimal dependencies for production use.

## Quick Start

### Building Images

```bash
# Build development image
./docker-run.sh build-dev

# Build release image  
./docker-run.sh build-release
```

### Running Simulations

```bash
# Run a simulation with the release image
./docker-run.sh run equil.txt

# Start interactive development environment
./docker-run.sh shell

# Start persistent development container
./docker-run.sh dev
```

## Development Workflow

1. Build the development image:
   ```bash
   ./docker-run.sh build-dev
   ```

2. Start a development shell:
   ```bash
   ./docker-run.sh shell
   ```

3. Inside the container, build Smoldyn:
   ```bash
   cd /workspace/smoldyn-gpu-gladkov
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j$(nproc)
   ```

4. Debug with GDB:
   ```bash
   gdb ./SmoldynGPU
   (gdb) run equil.txt
   ```

## Production Deployment

The release image is optimized for production use:

```bash
# Build release image
./docker-run.sh build-release

# Run simulation
./docker-run.sh run config.xml
```

## Available Commands

- `build-dev`: Build development Docker image
- `build-release`: Build release Docker image  
- `run <config>`: Run simulation with release image
- `dev`: Start persistent development container
- `shell`: Interactive development shell
- `clean`: Remove all Docker resources

## Requirements

- Docker with GPU support (nvidia-docker)
- NVIDIA drivers compatible with CUDA 11.6.1
- X11 forwarding for graphics (Linux/WSL)

## TODO
- [ ] Currently CUDA architecture is fixed to `sm_89` in CMakeLists. We need to allow targeting other architectures without having to change cmake files.
- [ ] Rename nvRadixSort to a non-misleading name (currently it is thrust sort, which is comparison based).
