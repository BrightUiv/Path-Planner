# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Genesis is a universal physics platform for Robotics/Embodied AI/Physical AI applications. It integrates multiple physics solvers (Rigid body, MPM, SPH, FEM, PBD, Stable Fluid) into a unified framework with photo-realistic rendering capabilities. The codebase is built on GsTaichi (a custom fork of Taichi) for cross-platform GPU acceleration and uses PyTorch for tensor operations.

## Development Commands

### Installation
```bash
# Development install (editable mode)
pip install -e ".[dev]"

# After switching branches/pulling updates, re-run to refresh dependencies and entrypoints
pip install -e ".[dev]"

# Install pre-commit hooks for automatic code formatting
pip install pre-commit
pre-commit install
```

### Testing
```bash
# Run required tests (minimal gate check before merging)
pytest -v --forked -m required ./tests

# Run full test suite (parallel execution by default)
pytest ./tests

# Run specific test markers
pytest -m "not slow" ./tests          # Skip slow tests
pytest -m benchmarks ./tests          # Run benchmarks only
pytest -m examples ./tests            # Run example tests only
```

### Code Formatting
- Black formatter is configured (line length 120)
- Automatically runs via pre-commit hooks
- Avoid editing `genesis/ext/` (contains external/vendor code, excluded from formatting)

### CLI Utilities
```bash
# View URDF/mesh files
gs view path/to/robot.urdf

# Create animation from rendered images
gs animate "renders/*.png" --fps 30
```

### Docker
```bash
# Build Docker image
docker build -t genesis -f docker/Dockerfile docker

# For AMD GPUs
docker build -t genesis-amd -f docker/Dockerfile.amdgpu docker
```

## Code Architecture

### Core System Initialization

Genesis uses a global initialization pattern via `gs.init()` which configures:
- **Backend selection**: CPU, CUDA, Vulkan, or Metal (platform-dependent)
- **Precision**: 32-bit or 64-bit floating point
- **GsTaichi configuration**: Dynamic arrays, fast cache mode, zero-copy memory
- **Random seed**: For reproducible simulations
- **Logger**: Unified logging across Genesis and Taichi

The system MUST be initialized before creating scenes: `gs.init(backend=gs.gpu, precision="32")`. Call `gs.destroy()` to release GPU memory and cache compiled kernels.

### Scene-Entity-Solver Architecture

Genesis follows a hierarchical composition pattern:

1. **Scene** (`genesis/engine/scene.py`): Top-level container
   - Manages multiple entities, solvers, couplers, visualizers, and sensors
   - Provides the simulation loop via `scene.step()` or `scene.build()` + step iterations
   - Tracks simulation state (`SimState`) across all solvers
   - Handles reset/destroy lifecycle

2. **Entities** (`genesis/engine/entities/`): Physical objects in the simulation
   - Base class: `BaseEntity` defines common interface
   - Specialized entities per solver:
     - `RigidEntity`: Articulated robots, rigid objects (URDF/MJCF loading)
     - `DroneEntity`: Specialized rigid entity with aerodynamic forces
     - `MPMEntity`: Material Point Method (liquids, sand, snow, deformable solids)
     - `SPHEntity`: Smoothed Particle Hydrodynamics (fluids)
     - `FEMEntity`: Finite Element Method (soft bodies, cloth)
     - `PBDEntity`: Position-Based Dynamics (cloth, deformable objects)
     - `SFEntity`: Stable Fluid (smoke, gases)
     - `HybridEntity`: Combines multiple material types
     - `ToolEntity`: Differentiable cutting/manipulation tools
     - `AvatarEntity`: Human avatars with speech/motion

3. **Solvers** (`genesis/engine/solvers/`): Physics simulation backends
   - Each solver type handles entities of corresponding material
   - `RigidSolver`: Constraint-based articulated body dynamics (decomposition-based collision detection)
   - `MPMSolver`, `SPHSolver`, `FEMSolver`, `PBDSolver`, `SFSolver`: Particle/continuum methods
   - Solvers are automatically instantiated when adding entities to a scene
   - All implemented in Taichi for GPU acceleration

4. **Couplers** (`genesis/engine/couplers/`): Inter-solver interaction
   - `LegacyCoupler`: Standard coupling between different physics solvers
   - `IPCCoupler`: Incremental Potential Contact for high-quality contact
   - `SAPCoupler`: Sweep-and-Prune for efficient broad-phase collision

5. **Materials** (`genesis/engine/materials/`): Define physical properties
   - Organized by solver type (MPM/, SPH/, FEM/, PBD/, SF/)
   - Examples: `Liquid`, `Snow`, `Sand`, `Elastic`, `Cloth`, `Smoke`
   - Each material specifies constitutive models and solver parameters

### Sensors and Rendering

- **Sensors** (`genesis/engine/sensors/`): Attach to entities for observations
  - `Camera`, `DepthCamera`: RGB and depth imaging
  - `IMU`: Inertial measurement
  - `ContactForce`: Force/torque sensing
  - `Raycaster`: Distance measurements

- **Renderers** (`genesis/options/renderers.py`):
  - `Rasterizer`: Fast OpenGL-based rendering via PyRender
  - `RayTracer`: Photo-realistic rendering via LuisaRender
  - `BatchRenderer`: High-throughput rendering via Madrona (Linux x86_64 only)

### State Management

The `SimState` object (`genesis/engine/states/solvers.py`) provides batched access to simulation state:
- Positions, velocities, forces for all entities
- Organized by solver type
- Supports batched environments for parallel training

### Examples Organization

- `examples/drone/`: Drone control and RL training (PPO, MAPPO for multi-agent)
- `examples/manipulation/`: Robotic manipulation tasks
- `examples/locomotion/`: Legged robot locomotion
- `examples/rendering/`: Rendering demonstrations
- `examples/tutorials/`: Getting started examples
- `examples/sensors/`: Sensor usage examples
- `examples/coupling/`: Multi-physics coupling demonstrations
- `examples/rigid/`: Rigid body dynamics examples

## Coding Conventions

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Python version**: Requires `>=3.10,<3.14`
- **Indentation**: 4 spaces
- **Line length**: 120 characters (Black formatter)
- **Type hints**: Use pydantic models for options/configurations

## Pull Request Guidelines

- Prefix PR titles with: `[BUG FIX]`, `[FEATURE]`, `[MISC]`, or `[CHANGING]` (breaking changes)
- Include test commands or minimal reproduction script in PR description
- PRs require at least one approval and passing CI before merging
- Use "Squash and Merge" for clean commit history

## Important Technical Details

### Backend Considerations

- **Metal backend**: May be unstable, doesn't support 64-bit precision or dynamic arrays
- **Vulkan backend**: Use `gs.init(backend=gs.vulkan)` for AMD GPUs
- **Performance mode**: Set `performance_mode=True` in `gs.init()` to disable dynamic arrays for maximum speed
- **Zero-copy**: Automatically enabled for CPU/CUDA backends when using dynamic arrays

### Entity Loading

Entities can be loaded from various formats:
- Robots: URDF (`.urdf`), MJCF (`.xml`)
- Meshes: `.obj`, `.stl`, `.glb`, `.ply`, `.dae`
- Assets location: `genesis/assets/` contains built-in models

When adding entities, you must call `scene.build()` before stepping the simulation.

### Parallel Environments

Genesis supports massive parallelization:
- Create batched scenes via `n_envs` parameter in `Scene` constructor
- Access batched states via indexing: `scene.entities[i].get_state()[env_idx]`
- Ideal for RL training (used with RSL-RL, IsaacGym-style workflows)

### Differentiability

Currently supported in:
- MPM solver (gradients through particle simulations)
- Tool solver (differentiable cutting)
- Rigid body solver (planned for future releases)

Access gradients via Taichi's autodiff capabilities when enabled.

## Documentation

- Official docs: https://genesis-world.readthedocs.io/en/latest/ (English, Chinese, Japanese)
- Report bugs: https://github.com/Genesis-Embodied-AI/Genesis/issues
- Discussions: https://github.com/Genesis-Embodied-AI/Genesis/discussions
