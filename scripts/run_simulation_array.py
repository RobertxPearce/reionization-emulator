# ------------------------------------------------------------------------------------------
# Run Reionization Simulations Using Latin Hypercube Sampling
# Robert Pearce
# ------------------------------------------------------------------------------------------

import os
import sys
from pathlib import Path
import subprocess as sp
import numpy as np


# Path to simulation executable and output directory
EXEC = Path("/jet/home/rpearce/software/ksz_2lpt/ksz_2lpt.x")
OUT  = Path("~/ocean/raw/sims_v6").expanduser()

# Path to the txt parameter samples
PARAM_FILE = Path("~/ocean/raw/param_samples/params_v6.txt").expanduser()


def main() -> int:
    # Check output root exists
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Get array job id (--array=0-999)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    
    # Load the samples
    samples = np.loadtxt(PARAM_FILE, comments="#")
    # Load the row
    row = samples[task_id]
    
    # Create the output directory
    outdir = OUT / f"sim{task_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Output status
    print(f"Running Simulation: {task_id}")
    
    # Build the command ksz_2lpt expects: dir_out zmean_zre alpha_zre kb_zre b0_zre
    out_prefix = str(outdir) + "/"
    args = [str(EXEC), out_prefix] + [str(v) for v in row]
    rc = sp.run(args).returncode
    
    # Output status
    print(f"Simulation {task_id} completed. Return code: {rc}")
    
    return rc


if __name__ == "__main__":
    sys.exit(main())

#-----------------------------
#         END OF FILE
#-----------------------------
