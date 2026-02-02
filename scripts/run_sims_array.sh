#!/bin/bash
#SBATCH -J ksz_2lpt
#SBATCH -p RM
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --ntasks=1            # One task per array element
#SBATCH --cpus-per-task=1     # One CPU core per task
#SBATCH --array=0-999%32      # Tasks 0-999, max 32 running at once
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH -A ast180004p
#SBATCH --mail-type ALL
#SBATCH --mail-user robertbdpearce@gmail.com

set -euo pipefail

# Intel
module unload intel
module load intel-icc
module load intel-mkl
which ifort

# Set environment variables
export KMP_LIBRARY=turnaround
export KMP_SCHEDULE=static
export KMP_STACKSIZE=256m
# Number of OpenMP threads (match with Slurm CPU request)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "Host=$(hostname)"
echo "PWD=$(pwd)"



# Define the simulation code directory
SIM_DIR="/jet/home/rpearce/software/ksz_2lpt/"
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
MAMBA="/jet/home/rpearce/miniforge3/bin/mamba"

# Go to simulation code directory
echo "Changing directory to $SIM_DIR for compilation"
cd $SIM_DIR

# Compile once per array job
LOCKFILE=".build.lock"
if mkdir "$LOCKFILE" 2>/dev/null; then
  echo "Compiling ksz_2lpt.x (lock acquired)"
  # Remove lock
  trap 'rmdir "$LOCKFILE"' EXIT
  make clean
  make ksz_2lpt.x
else
  echo "Another task is compiling."
  # Wait until the executable exists
  while [ ! -x "$SIM_DIR/ksz_2lpt.x" ]; do
    sleep 2
  done
fi

# Go to script directory for execution
echo "Changing directory to $SCRIPT_DIR for execution"
cd $SCRIPT_DIR

# Run Python inside the conda env
"$MAMBA" run -n simenv python "$SCRIPT_DIR/run_simulation_array.py"

