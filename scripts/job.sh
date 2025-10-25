#!/bin/bash
#SBATCH -J ksz_2lpt
#SBATCH -p RM
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 128
#SBATCH -o std.log
#SBATCH -A ast180004p
#SBATCH --mail-type ALL
#SBATCH --mail-user robertbdpearce@gmail.com

# Intel
module unload intel
module load intel-icc
module load intel-mkl
which ifort

# Set environment variables
export KMP_LIBRARY=turnaround
export KMP_SCHEDULE=static,balanced
export KMP_STACKSIZE=256m

# Go to the job scratch directory
# cd $SLURM_SUBMIT_DIR

# Define the simulation code directory
SIM_DIR = "/jet/home/rpearce/software/ksz_2lpt/"
SCRIPT_DIR = $SLURM_SUBMIT_DIR

# Go to simulation code directory
echo "Changing directory to $SIM_DIR for compilation"
cd $SIM_DIR

# Clean and compile the simulation code
make clean
make ksz_2lpt.x

# Go to script directory for execution
echo "Changing directory to $SCRIPT_DIR for execution"
cd $SCRIPT_DIR

# Activate environment
mamba activate simenv

# Run just the simulation
# ./ksz_2lpt.x > log

# Run the Python script
python run_simulations.py

# Deactivate environment
conda deactivate
