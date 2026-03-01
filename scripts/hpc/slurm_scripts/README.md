# Available Job Scripts

- **`run_all_sims.sh`**  
    Runs all simulations sequentially within a single SLURM job. Requests cluster resources, compiles the simulation code, and executes a Python script.


- **`run_sims_array.sh`**  
    Uses a SLURM job array to run simulations in parallel, one parameter set per task. Requests cluster resources, compiles the simulation code, and executes a Python script.
