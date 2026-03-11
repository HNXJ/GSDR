# GSDR: Genetic Stochastic Delta Rule

Biophysical neural network modeling and optimization using JAX and Jaxley.

## 🚀 Core Notebooks
- **[Biophys_SX.ipynb](./Biophys_SX.ipynb)**: Main simulation and optimization pipeline for 3-population (E, IG, IL) cortical networks.
- **[kappa_synch.ipynb](./kappa_synch.ipynb)**: Detailed analysis of population synchrony using Fleiss' Kappa.

## 📦 Modular Library
The repository includes the `gsdr/` package for modular development:
- `gsdr.models`: Custom HH channels and synapses (AMPA, GABAa, GABAb).
- `gsdr.optimizers`: Implementation of GSDR and AGSDR v2.
- `gsdr.analysis`: Spectral and synchrony analysis tools.
- `gsdr.data_loader`: Standardized loading of biological comparison data.

## 🧬 Biological Data
The model is validated against the `oxm0818` dataset. Raw unit data is required for PSD and Kappa validation plots.
- **Path**: `drive/ReadOnly/`
- **Files**:
  - `oxm0818_units.npy`: Continuous traces for sorted units.
  - `oxm0818_units_info.npy`: Metadata for units.

*Note: Data files are located in the synced Google Drive ReadOnly directory.*
