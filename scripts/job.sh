#!/bin/bash
#SBATCH -J ksz_2lpt
#SBATCH -p RM
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 128
#SBATCH -o std.log
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

# Define the simulation code directory
SIM_DIR="/jet/home/rpearce/software/ksz_2lpt/"
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
KSZ_BIN="$SIM_DIR/ksz_2lpt.x"
MAMBA="/jet/home/rpearce/miniforge3/bin/mamba"

# Go to simulation code directory
echo "Changing directory to $SIM_DIR for compilation"
cd $SIM_DIR

# Clean and compile the simulation code
make clean
make ksz_2lpt.x

# Go to script directory for execution
echo "Changing directory to $SCRIPT_DIR for execution"
cd $SCRIPT_DIR

# Run Python inside the conda env
"$MAMBA" run -n simenv python "$SCRIPT_DIR/run_simulations.py"

# Activate environment
# mamba activate simenv

# Run the Python script
# python run_simulations.py

# Deactivate environment
# conda deactivate
