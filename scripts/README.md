# Scripts

Scripts for parameter sampling, HPC simulation runs, and building the training dataset. These are **environment-specific**: paths and cluster settings (e.g. SLURM, Fortran binary location) may need to be adapted for your system.

The **core library** is in `src/reionemu/` and is pip-installable; these scripts show how to use it in practice.

---

## sampling/

- **[lhs_one_param.py](sampling/lhs_one_param.py)**  
  Latin Hypercube sampling (LHS) for one reionization parameter while holding the others fixed.

- **[lhs_four_params.py](sampling/lhs_four_params.py)**  
  LHS for four reionization parameters (zmean, alpha, kb, b0). Outputs a parameter file (one sample per row).

---

## hpc/

- **[run_simulations.py](hpc/run_simulations.py)**  
  Cluster runner (single-job batch): generates (zmean, alpha, kb, b0) via LHS and executes the Fortran simulation for each sample. Typically launched from a SLURM submission script.

- **[run_simulation_array.py](hpc/run_simulation_array.py)**  
  SLURM array runner: reads a parameter file and runs the row indexed by SLURM_ARRAY_TASK_ID.

### hpc/slurm_scripts/
- **[run_sims_array.sh](hpc/slurm_scripts/run_sims_array.sh)**  
  sbatch array script (one parameter set per task).
- **[run_all_sims.sh](hpc/slurm_scripts/run_all_sims.sh)**  
  Single-job batch script (runs many simulations in one allocation).

---

## dataset/

- **[build_training_h5.py](dataset/build_training_h5.py)**  
  Builds a training-ready HDF5 dataset from simulation outputs (inputs = parameters, targets = angular power spectra).