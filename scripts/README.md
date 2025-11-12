# Available Scripts

- **`run_simulations.py`**  
    Generates parameters zmean, alpha and kbar using Latin Hypercube. Then runs the Fortran simulation executable.


- **`build_dataset.py`**  
    Creates a dataset for the emulator by organizing the parameters and output into condensed HDF5 file.

    ```
    Top-Level:
        ['sims']
    Header:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    Params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    Output:
        ['ksz_map', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    ```

- **`compute_cl.py`**  
    Calculates the angular power spectrum (C_ell) from each kSZ map using either a manual flat-sky FFT method or the powerbox.get_power() fuinction. The resulting spectra (C_ell and D_ell) are stored in the processed HDF5 file under each simulation's /cl group.

    ```
    Top-Level:
        ['sims']
    Header:
        ['sim0'], ['sim1'], ['sim2'], ... , ['sim<n>']
    sim<n>:
        [params], [output], [cl]
    Params:
        ['alpha_zre', 'b0_zre', 'kb_zre', 'zmean_zre']
    Output:
        ['ksz_map', 'pk_tt', 'tau', 'xmval_list', 'zval_list']
    Cl:
        ['cl_ksz', 'dcl', 'dl_ksz', 'ell']
    ```

- **`job.sh`**  
    A SLURM submission script that requests cluster resources, compiles the simulation code, and executes teh Python script.
