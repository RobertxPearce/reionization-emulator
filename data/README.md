# Data Directory

This folder organizes all simulation outputs, intermediate products, and final datasets used for the emulator.

## Structure

- **`param_samples/`** – Param samples generated using Latin Hypercube Sampling.   
    Format: Text files


- **`raw/`** – Raw outputs directly from the Zerion Fortran simulations.  
    Format: HDF5 files


- **`processed/`** – Processed HDF5 datasets containing model parameters, kSZ maps, angular power spectra (C_ell and D_ell) for each simulation, organized under /sims. Training data is saved under /training and contains X, Y, ell, sim ids, param names, and meta data for emulator training.  
    Format: HDF5 files

## Data Access
[Google Drive Folder](https://drive.google.com/drive/folders/1pRjommf3gfslKHRcas5jHkyMgf0knkXc?usp=sharing)
