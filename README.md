# Reionization Emulator
This project explores how the universe's first galaxies reionized the intergalactic medium by modeling the **kinetic Sunyaev-Zel'dovich (kSZ)** effect during the **Epoch of Reionization (Eor)**.
I'm working under [Dr. Paul La Plante](https://plaplant.github.io/) at UNLV's LEADS Lab, using the **Bridges-2 supercomputer** at the Pittsburgh Supercomputing Center (PSC) to run large-scale cosmological simulation with the *Zerion* model.

The main goal of this project is to **build a machine learning emulator** that predicts the **kSZ angular power spectrum (Cl)** directly from reionization model parameters. This would allow for rapid exploration of cosmological parameter space without running new simulations.

## Scientific Background
The **kSZ effect** measures distortions in the Cosmic Microwave Background (CMB) caused by free electrons moving during reionization.
By simulating this process and computing the **angular power spectrum**, we can extract statistical information about when and how quickly reionization occurred.

## Workflow Overview
1. **Run Zerion simulations** on Bridges-2 HPC with varying reionization parameters using Latin Hypercube Sampling (`z_mean`, `alpha_zre`, ` kb_zre`, `b0_zre`).
2. **Build dataset** containing model parameters and select simulation outputs formated as HDF5 files
    - Model Parameters
      - `alpha_zre` - Controls how long reionization lasts. Bounded: [0.10 - 0.90] 
      - `kb_zre` Determines how uneven the ionized regions are (higher values more clustered). Bounded: [0.10 - 2.0]
      - `z_mean` Sets the midpoint of reionization. Bounded: [7.0 - 9.0]
      - `b0_zre` Adjusts the overall strength or amplitude of the ionization field. Constant:
    - **Simulation Output**
      - `tau` - The optical depth to reionization; measures how many CMB photons were scattered by free electrons.
      - `ksz_map` - A 2D map of temperature fluctuations caused by the kinematic Sunyaev–Zel’dovich effect.
      - `pk_tt` - The 3D matter power spectrum P(k); shows how matter is distributed across different spatial scales within the simulation volume.
      - `xmval_list` - The average ionized fraction of the universe at different times; shows how reionization progressed.
      - `zval_list` - The redshift values that correspond to each point in the ionization history.
3. **Compute angular power spectrum** from each kSZ map using a 2D Fourier transform and add data into processed dataset.
   - `ell`- The angular frequency
   - `cl_ksz`- The raw angular power spectrum in uK^2 (variance per angular scale)
   - `dcl`- Basic uncertainty per bin
   - `dcl_ksz`- The rescaled spectrum
4. **Train emulator** to predict angular power spectrum given new parameters using a neural network.

## Directory Structure
- `src` - Directory for the emulator architecture, training logic, model evaluation and data preprocessing.
- `data` - Directory containing the direct simulation output (raw), processed simulation data (processed), and test sets.
- `scripts` - Python scripts for running simulations, processing data, computing angular power spectrum.
- `notebooks` - Jupyter notebooks for analysis and visualization.
- `results` - Plots and charts documenting progress.

## Acknowledgments
This work is part of ongoing research with [Dr. Paul La Plante](https://plaplant.github.io/) in the LEADS Lab at the [University of Nevada, Las Vegas](https://www.unlv.edu/cs), using computing resources provided by the Pittsburgh Supercomputing Center (PSC) through the ACCESS program.