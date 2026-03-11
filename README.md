# GSDR: Genetic Stochastic Delta Rule

Biophysical neural network modeling and optimization using JAX and Jaxley.

## 🚀 Core Notebooks
- **[Biophys_SX.ipynb](./Biophys_SX.ipynb)**: Main simulation and optimization pipeline for EI network example.
- **[kappa_synch.ipynb](./kappa_synch.ipynb)**: Detailed analysis of population synchrony using Fleiss' Kappa.

## 📦 Modular Library
The repository includes the `gsdr/` package for modular development:
- `gsdr.models`: Custom HH channels and synapses (AMPA, GABAa, GABAb).
- `gsdr.optimizers`: Implementation of GSDR and AGSDR v2.
- `gsdr.analysis`: Spectral and synchrony analysis tools.
- `gsdr.data_loader`: Standardized loading of biological comparison data.

## 🧬 Neurophysiology Data
The data used for electrophysiologic analysis parts :
- **Public Data Link**: [Google Drive - oxm0818 Dataset](https://drive.google.com/drive/folders/1TwEl4AERajbhQe8-kFExx8EcCLMAZh2O?usp=sharing)
- **Included Files**: Sorted units based on Multi-area Dense Laminar Neurophysiology (MaDeLaNe)
  - `oxm0818_units.npy`: Continuous traces for sorted units (from three 128-channel probes)
  - `oxm0818_units_info.npy`: Metadata for units.
  - For more information regarding the experimental design and recording details, see the Methods section of manuscript
