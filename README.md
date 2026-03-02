# kSZ 2LPT Reionization Emulator

Machine learning emulator for the kinetic Sunyaev–Zel’dovich (kSZ) angular power spectrum during the Epoch of Reionization (EoR).

This project combines:

- Large-scale cosmological simulations (Zreion model)
- High-performance computing (Bridges-2, PSC)
- Deterministic power spectrum computation
- A modular Python emulator library
- Neural network surrogate modeling

The goal is to learn a fast surrogate model that maps reionization parameters $\rightarrow$ binned kSZ power spectrum, enabling rapid exploration of cosmological parameter space without re-running expensive simulations.

This work is conducted under [Dr. Paul La Plante](https://plaplant.github.io/) in the LEADS Lab at UNLV.

---

## Scientific Context

The kinetic Sunyaev–Zel’dovich (kSZ) effect is produced when CMB photons undergo Thomson scattering from free electrons with bulk velocities, leading to Doppler-induced temperature fluctuations.

From each simulation, we compute the angular power spectrum:


$D_\ell = \frac{\ell(\ell+1)}{2\pi} C_\ell $ 

These spectra contain statistical information about:

- The timing of reionization
- Its duration
- The clustering of ionized regions

The emulator approximates this forward mapping.

---

## Problem Formulation

We learn a mapping: $z_{mean}$ , $\alpha$ , $k_b$ , $b_0$ $\rightarrow$ $D_\ell $

### Input Parameters

| Parameter | Description | Bounds |
|------------|------------|--------|
| `alpha_zre` | Controls reionization duration | [0.10, 0.90] |
| `kb_zre` | Controls clustering of ionized regions | [0.10, 2.0] |
| `z_mean` | Midpoint redshift of reionization | [7.0, 9.0] |
| `b0_zre` | Overall ionization amplitude | [0.10, 0.80] |

---

# Emulator Library Architecture

The `src/` directory contains a purpose-built, modular Python library designed to make the full pipeline reproducible and extensible.

The library cleanly separates:

### 1. Simulation I/O (`simio/`)
- Condenses raw Zreion outputs into a structured HDF5 layout
- Computes flat-sky angular power spectra from kSZ maps
- Builds ML-ready training arrays `(X, Y)` with optional log transforms

### 2. Data Utilities (`data/`)
- Deterministic feature standardization
- Dataset wrappers and PyTorch DataLoaders
- Explicit parameter ordering and metadata tracking

### 3. Models (`models/`)
- Minimal MLP baselines (3- and 4-parameter variants)
- Designed for easy extension to deeper architectures or Bayesian models

### 4. Training (`training/`)
- Reusable PyTorch training loop
- Early stopping support
- Device-aware execution (MPS/CPU/GPU)
- Configurable gradient clipping

### Design Goals

- Reproducible HDF5 pipeline
- Explicit parameter ordering
- Deterministic preprocessing
- Minimal coupling between simulation and ML code
- Research-friendly extensibility (e.g., BNNs)

The emulator library allows experiments to be run from scripts or notebooks without rewriting preprocessing or training logic.

---

## Workflow

### 1. Parameter Sampling
Latin Hypercube Sampling across the 4D parameter space.

### 2. Simulation (HPC)
Zreion simulations are executed on Bridges-2 using SLURM job arrays.

### 3. Dataset Construction
- Condense raw HDF5 outputs
- Compute flat-sky angular power spectra
- Bin spectra into fixed ℓ bins
- Build training arrays (X, Y)

### 4. Emulator Training
- Standardize inputs/targets
- Train baseline neural networks
- Evaluate MSE on validation splits

---

## Repository Structure

```bash
└── reionization-emulator/
    ├── src/
    │   └── emulator/       Core emulator library
    │       ├── simio/      Simulation I/O + power spectrum computation
    │       ├── data/       Standardization + dataloaders
    │       ├── models/     PyTorch model architectures
    │       └── training/   Reusable training loop and K-Fold Clustering
    ├── notebooks/          Analysis and validation notebooks
    ├── data/               Raw and processed simulation data (not tracked)
    ├── scripts/            HPC runners, dataset builders, sampling utilities
    ├── results/            Figures and experiment outputs
    └── checkpoints/        Saved model weights + normilization artifacts
```

---

## Installation

Install the emulator package in editable mode:

```bash
python -m pip install -e .
```

---

## Long-Term Goals
- Replace expensive simulations with a fast surrogate 
- Enable likelihood-based cosmological parameter inference 
- Incorporate uncertainty quantification (e.g., Bayesian neural networks)
- Scale beyond proof-of-concept MLP architectures

---

## Acknowledgments
This research is conducted in the LEADS Lab at the University of Nevada, Las Vegas, under [Dr. Paul La Plante](https://plaplant.github.io/), using computing resources from the Pittsburgh Supercomputing Center (Bridges-2).
