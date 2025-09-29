from pathlib import Path
import subprocess as sp
from scipy.stats import qmc


# Path to simulation executable and output directory
EXEC = Path("/jet/home/rpearce/software/ksz_2lpt/ksz_2lpt.x")
OUT  = Path("~/ocean/emulatorv0").expanduser()

# Number of samples / simulations
NUM_SAMPLES = 10
# Seed for reproducible results
SEED = 123

# Params and their bounds
PARAMS = [
    ("zmean", 6.0, 10.0),
    ("alpha", 0.10, 0.50),
    ("kb",    1.0, 10.0),
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


def run_samples(samples, outroot):
    """
    Run the simulation once for each row of sampled parameters
    samples: 2D array where each row is one parameter set
    outroot: Path to the base output directory; subfolders sim0, sim1, are created
    """
    # Open a log file in the base output directory
    log_path = outroot / "sim_log.txt"
    with log_path.open("w") as log:
        # Write header line
        log.write("sim_id\tzmean\talpha\tkb\n")

        # Loop over all rows of the samples array
        for i, row in enumerate(samples):
            # Make a new folder for the simulation i
            outdir = outroot / f"sim{i}"
            outdir.mkdir(parents=True, exist_ok=True)
            # Build the command
            args = [str(EXEC), str(outdir)] + [str(v) for v in row]
            # Run the executable
            rc = sp.run(args).returncode
            # Write run info to log (tab-separated)
            log.write(f"sim{i}\t{row[0]}\t{row[1]}\t{row[2]}\n")
            # Print progress: run number + return code
            print(f"[{i+1}/{len(samples)}] rc={rc}")


def main():
    # Check if output directory exists / create it
    OUT.mkdir(parents=True, exist_ok=True)
    # Take just the lo and hi bounds of each param
    bounds = [(lo, hi) for (_name, lo, hi) in PARAMS]
    # Sample parameter sets with Latin Hypercube Sampling
    samples = latin_hypercube(bounds, NUM_SAMPLES, SEED)
    # Run one simulation per sample
    run_samples(samples, OUT)


if __name__ == "__main__":
    main()
