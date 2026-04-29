# Datasets Directory

This folder organizes all simulation outputs, intermediate products, and final datasets used for the emulator. HDF5 data files and processed outputs are not tracked by git; parameter sample files and simulation logs are tracked.

## Structure

- **param_samples/** – Param samples generated using Latin Hypercube Sampling.   
    Format: Text files


- **raw/** – Raw outputs directly from the Zreion Fortran simulations.
    Format: HDF5 files


- **processed/** – Processed HDF5 datasets containing model parameters, kSZ maps, angular power spectra (C_ell and D_ell) for each simulation, organized under /sims. Training data is saved under /training and contains X, Y, ell, sim ids, param names, and metadata for emulator training.  
    Format: HDF5 files
