# CDM-SIDM-Subhalo-Analysis
This repository contains a pipeline for analyzing dark matter subhalos extracted from cosmological N-body simulations.  
The code computes structural properties, generates density profiles, and performs parametric fits for both Cold Dark Matter (CDM) and Self-Interacting Dark Matter (SIDM) scenarios.

---

## Overview

The main goals of this project are:
- Identify and select well-resolved and appropriate subhalos from simulation outputs
- Compute radial density profiles from particle data
- Fit analytical halo models (NFW and isothermal profiles)
- Visualize subhalo structure and fit quality

---

## Features

- HDF5 snapshot parsing (`h5py`)
- Subhalo selection and quality cuts:
  - Centering constraints
  - Relaxation criterion
  - Virial ratio check
- Radial binning and density estimation
- Nonlinear curve fitting (`scipy.optimize.curve_fit`)
- SIDM / CDM analysis modes
- Automated figure generation

## Example Output for CDM with NFW Fit
- Relaxed halo
- More than minimum number of particle
- Fitted NFW profile outside the trust radius
<img width="500" height="500" alt="subhalo533" src="https://github.com/user-attachments/assets/2405c54a-93bf-422a-a3bb-479d0d16fef9" />
<img width="500" height="500" alt="subhalo533_density" src="https://github.com/user-attachments/assets/a7753146-7294-44c9-8e01-3df7a73f1fed" />

## References
- Spergel & Steinhardt (2000), Phys. Rev. Lett. 84, 3760
- Springel (2010), MNRAS 401, 791
- Weinberger, Springel & Pakmor (2020), ApJS 248, 32
- Vogelsberger, Zavala & Loeb (2012), MNRAS 423, 3740
