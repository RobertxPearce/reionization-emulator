# ------------------------------------------------------------------------------------------
# Sample a single reionization parameter using Latin Hypercube Sampling
# Robert Pearce
# ------------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import qmc

# Params and their bounds
PARAMS = [
    ("zmean", 7.0, 9.0),
    ("alpha", 0.10, 0.90),
    ("kb",    0.10, 2.0),
    ("b0", 0.10, 0.80), # Updated From [0.40, 2.0]
]


def lhs_one_param(params, sample_name, n, seed=0, fixed=None):
    """
    LHS-sample One Parameter; keep all others fixed

    fixed: Dict of {name: value} for non-sampled params. If not provided,
           it uses midpoint of that param's bounds.
    Returns: List of dicts [{param: value}, ...}, ...]
    """
    fixed = fixed or {}

    # Look up bounds
    bounds = {name: (lo, hi) for name, lo, hi in params}
    # Check if sample_name is correct
    if sample_name not in bounds:
        raise ValueError(f"Sample name {sample_name} not in bounds")
    # Set bounds
    lo, hi = bounds[sample_name]

    # 1D LHS for the chosen parameter
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n)
    vals = qmc.scale(u, [lo], [hi]).flatten()

    # Build the param dicts
    out = []
    for v in vals:
        d = {}
        for name, lo_i, hi_i in params:
            if name == sample_name:
                d[name] = float(v)
            else:
                d[name] = float(fixed.get(name, 0.5 * (lo_i + hi_i)))
        out.append(d)

    return out


def print_table(runs):
    """
    Helper function for printing table of parameter sampling
    """
    headers = runs[0].keys()
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * (13 * len(headers)))

    for r in runs:
        print(" | ".join(f"{r[h]:10.4f}" for h in headers))


def main():
    # Sample one parameter with fixed values for the other parameters
    runs = lhs_one_param(params=PARAMS, sample_name="b0", n=4, seed=3, fixed={"zmean": 7.8739127780643505, "alpha": 0.2292734227025891, "kb": 1.5829409586172485})

    # Sample one parameter with other params defaulting to midpoint
    # runs = lhs_one_param(params=PARAMS, sample_name="b0", n=4, seed=42, fixed=None)

    print_table(runs)



if __name__ == "__main__":
    main()

#-----------------------------
#         END OF FILE
#-----------------------------
