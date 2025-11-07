# Available Scripts

- **`run_simulations.py`**  
    Generates parameters zmean, alpha and kbar using Latin Hypercube. Then runs the Fortran simulation executable.


- **`build_dataset.py`**  
    Creates an HDF5 dataset for the emulator by collecting ksz_map, pk_tt, xmval_list, zval_list, alpha_zre, kb_zre, and zmean_zre from simulation outputs.


- **`compute_cl.py`**  
    Calculates the angular power spectrum (C_ell) from each kSZ map using either a manual flat-sky FFT method or the powerbox.get_power() fuinction. The resulting spectra (C_ell and D_ell) are stored in the processed HDF5 file under each simulation's /cl group.


- **`job.sh`**  
    A SLURM submission script that requests cluster resources, compiles the simulation code, and executes teh Python script.
