# ------------------------------------------------------------------------------------------
# Sample four reionization parameters (zmean, alpha, kb, and b0) using Latin Hypercube 
# Sampling and save params to txt file
# Robert Pearce
# ------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
from scipy.stats import qmc

# Path for .txt file containing all sampled params
OUT  = Path(r"/Users/robertxpearce/Desktop/reionization-emulator/data/param_samples")
# Name of file (version corresponding to the batch run)
FILE_NAME = "params_test.txt"

# Number of samples
NUM_SAMPLES = 5
# Seed for reproducible results
SEED = 702

# Params and their bounds
PARAMS = [
    ("zmean", 7.0, 9.0),
    ("alpha", 0.10, 0.90),
    ("kb",    0.10, 2.0),
    ("b0", 0.10, 0.80), # Updated From [0.40, 2.0]
]


def latin_hypercube(bounds, n, seed):
    """
    Return an (n x d) array of samples scaled to bounds using SciPy's LHS
    bounds: List of (lo, hi) for each parameter
    n: Number of samples to generate
    seed: Random seed
    """
    # Create the Latin Hypercube Sampler in d dimensions (number of params)
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    # Draw n samples in the unit hypercube [0,1]^d
    unit = sampler.random(n=n)
    # Separate the lower and upper bounds for each param
    lows  = [lo for (lo, hi) in bounds]
    highs = [hi for (lo, hi) in bounds]
    # Scale each column of params from [0,1] to [lo, hi]
    return qmc.scale(unit, lows, highs)


def main():
    # Check if output directory exists / create it
    OUT.mkdir(parents=True, exist_ok=True)
    # Take names and bounds
    names = [name for (name, _lo, _hi) in PARAMS]
    bounds = [(lo, hi) for (_name, lo, hi) in PARAMS]
    # Sample parameter sets with Latin Hypercube Sampling
    samples = latin_hypercube(bounds, NUM_SAMPLES, SEED)
    # Build output path
    out_path = OUT / FILE_NAME
    # Build header
    header_lines = [
        "Latin Hypercube Sampling Parameter Sets",
        f"Number of Samples: {NUM_SAMPLES}",
        f"Seed: {SEED}",
        "Columns =" + " ".join(names),
        "Bounds =" + " ".join([f"{n}[{lo},{hi}]" for (n, lo, hi) in PARAMS]),
        " ".join(names),
    ]
    header = "\n".join(header_lines)
    # Save as whitespace delimited text
    np.savetxt(out_path, samples, fmt="%.17g",delimiter=" ", header=header, comments="# ",)
    print(f"Wrote {NUM_SAMPLES} samples to: {out_path}")

if __name__ == "__main__":
    main()

#-----------------------------
#         END OF FILE
#-----------------------------
