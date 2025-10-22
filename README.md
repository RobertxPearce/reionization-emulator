# Reionization Emulator
I’m using the Bridges-2 supercomputer at the Pittsburgh Supercomputing Center (PSC) to run reionization simulations. These simulations produce kSZ (kinetic Sunyaev–Zel’dovich) maps, which I then analyze to extract the angular power spectrum (Cl) an important observable for studying the Epoch of Reionization.


As part of this work, I’m building an emulator that can quickly predict the kSZ power spectrum from reionization model parameters.


### Workflow
1. Run simulations on Bridges-2 supercomputer
2. Output of simulations HDF5 with kSZ map, Tau, ionization history
3. Compute power spectrum (Cl)
4. Train emulator on parameter-spectrum pairs
