# Available Notebooks

- **`data_vis.ipynb`**  
    Opens a single simulation’s raw HDF5 outputs (obs_grids.hdf5, pk_arrays.hdf5), inspects their structure and header values (including Tau), converts the kSZ map to μK, and visualizes it.


- **`proc_data_vis.ipynb`**  
    Loads the processed dataset containing all simulation outputs, summarizes key parameters and statistics, and visualizes relationships between the reionization parameters and optical depth (Tau) through histograms, scatter plots, and correlation analysis to identify parameter dependencies and trends.


- **`proof_of_concept.ipynb`**  
    Demonstrates a proof-of-concept emulator that predicts the binned kSZ angular power spectrum from reionization parameters and evaluates performance on held-out simulations.


- **`param_space_validation.ipynb`**  
    Validates coverage of the sampled reionization parameter space, ensuring the Latin Hypercube Sampling produces uniform and unbiased parameter coverage.
