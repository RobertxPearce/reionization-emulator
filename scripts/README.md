# Available Scripts

- **`run_simulations.py`**  
    Generates parameters zmean, alpha, kb and b0 using Latin Hypercube Sampling (LHS). Then runs the Fortran simulation executable.


- **`lhs_one_param.py`**  
    Generates Latin Hypercube samples for a single reionization parameter while keeping all other parameters fixed to constants or the midpoint.


- **`build_dataset.py`**  
    Creates a self-contained dataset for angular power spectrum calculation and emulator by organizing raw simulation parameters and outputs into a condensed HDF5 file.

    ```
    Top-Level:
        ['sims']
    sims:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    output:
        ['ksz_map', 'Tcmb0', 'theta_max_ksz', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    ```

- **`compute_cl.py`**  
    Calculates the angular power spectrum (C_ell) from each kSZ map using a manual flat-sky FFT method. The resulting spectra (C_ell and D_ell) are stored in the processed HDF5 file under each simulation's /cl group.

    ```
    Top-Level:
        ['sims']
    sims:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    output:
        ['ksz_map', 'Tcmb0', 'theta_max_ksz', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    cl:
        ['cl_ksz', 'dcl', 'dl_ksz', 'ell']
    ```

- **`job.sh`**  
    A SLURM submission script that requests cluster resources, compiles the simulation code, and executes teh Python script.
