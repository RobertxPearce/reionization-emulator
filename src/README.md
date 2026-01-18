# src Directory

- **`__init__.py`**  
    Marks the **src** directory as a Python package.


- **`preprocessing.py`**  
    Loads processed simulation data, extracts parameters and spectra, applies a log transform to the power spectrum, and builds the final emulator dataset (**params**, **log(d_ell)**, **ell**).


- **`emulator.py`**  
    Defines the neural network architecture used to predict the kSZ angular power spectrum from reionization parameters.


- **`train.py`**  
    Trains the emulator using the prepared dataset, performs normalization, and saves the trained model weights and normalization statistics.


- **`evaluate.py`**  
    Loads the trained model and normalization stats, evaluates performance on the test set, computes accuracy metrics, and plots true vs. predicted spectra.


## Current Metrics
