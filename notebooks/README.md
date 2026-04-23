# Notebooks

Analysis and training examples. The **primary package example** (full pipeline and API usage) is in the repo root: [../docs/reionemu_package_example.ipynb](../docs/reionemu_package_example.ipynb).

---

- **[param_space_validation.ipynb](param_space_validation.ipynb)**  
    Validates coverage of the sampled reionization parameter space, ensuring the Latin Hypercube Sampling produces uniform and unbiased parameter coverage.


- **[raw_data_vis.ipynb](raw_data_vis.ipynb)**  
    Opens a single simulation’s raw HDF5 outputs (obs_grids.hdf5, pk_arrays.hdf5), inspects their structure and header values (including Tau), converts the kSZ map to uK, and visualizes it.


- **[proc_data_vis.ipynb](proc_data_vis.ipynb)**  
    Loads the processed dataset containing all simulation outputs, summarizes key parameters and statistics, and visualizes relationships between the reionization parameters and optical depth (Tau) through histograms, scatter plots, and correlation analysis to identify parameter dependencies and trends.


- **[poc_three_param_model.ipynb](poc_three_param_model.ipynb)**  
    Demonstrates the experimental `POCEmulatorThreeParams` model that predicts the binned kSZ angular power spectrum from 3 reionization parameters ($\alpha$, $k_b$, $b_0$) and evaluates performance on held-out simulations.


- **[poc_four_param_model_kfold_cv.ipynb](poc_four_param_model_kfold_cv.ipynb)**  
    Uses K-Fold cross-validation to evaluate the four-parameter emulator model ($z_{mean}$, $\alpha$, $k_b$, $b_0$) for predicting the binned kSZ angular power spectrum.


- **[poc_four_param_model.ipynb](poc_four_param_model.ipynb)**  
    Demonstrates a proof-of-concept emulator that predicts the binned kSZ angular power spectrum from four reionization parameters and evaluates performance using a train/validation split.


- **[poc_mc_dropout_model.ipynb](poc_mc_dropout_model.ipynb)**
    This notebook trains a 4-parameter MC-dropout emulator for the binned kSZ angular power spectrum. The model predicts $\ln(D_\ell)$, then repeated dropout-enabled forward passes are used to estimate a predictive mean and uncertainty.

