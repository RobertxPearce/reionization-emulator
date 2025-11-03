# Available Scripts

- **`run_simulations.py`**  
    Generates parameters zmean, alpha and kbar using Latin Hypercube. Then runs the Fortran simulation executable.


- **`compute_cl.py`**  
    Calculates the angular power spectrum (Cl) from each kSZ map and saves the results for use in emulator training.


- **`build_dataset.py`**  
    Creates an HDF5 dataset for the emulator by collecting ksz_map, pk_tt, xmval_list, zval_list, alpha_zre, kb_zre, and zmean_zre from simulation outputs.


- **`job.sh`**  
    A SLURM submission script that requests cluster resources, compiles the simulation code, and executes teh Python script.
