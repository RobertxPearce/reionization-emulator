# Scripts

This directory contains helper scripts.

## Available Scripts

- **`run_simulations.py`**  
    Generates parameters zmean, alpha and kbar using Latin Hypercube. Then runs the Fortran simulation executable.


- **`compute_cl.py`**  
    Calculates the angular power spectrum (Cl) from each kSZ map and saves the results for use in emulator training.


- **`build_dataset.py`**  
    Collects simulation outputs, extracts kSZ maps, tau, and power spectra, and organizes them with input parameters into a training dataset for the emulator.

- **`job.sh`**  
    A SLURM submission script that requests cluster resources, compiles the simulation code, and executes teh Python script.