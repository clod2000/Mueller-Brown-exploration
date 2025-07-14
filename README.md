# Mueller-Brown-exploration

This repository contains a comprehensive framework for simulating and analyzing particle dynamics on the Müller-Brown potential—a prototypical model energy landscape widely used in computational chemistry and statistical physics. The project implements advanced sampling techniques, parallelization strategies, and in-depth data analysis to explore the equilibrium and dynamical properties of this potential.

## Key Features

- **Langevin and Metropolis Dynamics Integration**  
  The codebase supports both standard overdamped Langevin dynamics and its augmentation with a Metropolis acceptance step. This allows for robust sampling of the equilibrium distribution and detailed balance, with the option to compare pure Langevin to Metropolis-corrected dynamics.

- **Parallelized Simulation**  
  Trajectory generation is parallelized using Python's multiprocessing, enabling efficient simulation of many independent trajectories across multiple CPU cores. This accelerates large-scale statistical analyses.

- **Flexible Potential Functions**  
  Includes implementations of both the Müller-Brown potential and a 2D Gaussian potential for benchmarking.

- **Efficient Density Estimation**  
  Utilizes a parallelized kernel density estimation (KDE) routine for analyzing the sampled points, enabling fast computation of configuration space densities and corresponding free energies.

- **Comprehensive Analysis and Visualization**  
  Jupyter notebooks and Python scripts are provided for exploring simulation results, comparing sampled energy distributions to the true potential landscape, visualizing density maps, and assessing normalization factors.

## What I Did

- **Integrated Metropolis Correction:**  
  Developed a Metropolis-Hastings step compatible with Langevin proposals to rigorously enforce detailed balance in the sampling process.

- **Parallelized the Dynamics:**  
  Implemented parallel simulation of many independent Langevin/Metropolis trajectories using Python multiprocessing, allowing large statistical ensembles to be generated efficiently.

- **Automated Analysis Pipeline:**  
  Created Jupyter notebooks for loading, filtering, and visualizing data, including energy comparison, normalization checks, and density mapping.

- **Robust and Modular Code:**  
  Wrote modular utilities for potential functions, plotting, gradient computation, and KDE, making the codebase extensible and well-documented.

## Code Structure

- `langevin.py`: Main script for running overdamped Langevin dynamics (with or without Metropolis correction), saving data, and producing core visualizations. Supports parallel execution.
- `utils.py`: Library of functions for potential definitions (Müller-Brown, Gaussian), plotting, gradient computation, and parallelized density estimation.
- `analysis.ipynb`: Jupyter notebook for detailed post-processing, including energy normalization, density comparison, and visualization of the results.

## How to Use

1. **Dependencies**  
   - Python 3.x  
   - Libraries: `numpy`, `torch`, `matplotlib`, `scipy`, `pandas`, `joblib` (for parallelization)

2. **Run Simulations**  
   - Edit parameters in `langevin.py` as needed (potential type, Metropolis toggle, number of cores, etc.).
   - Execute `langevin.py` to generate trajectory data and figures. Data and results are saved in structured folders.

3. **Analyze Data**  
   - Open `analysis.ipynb` to load simulation results, filter configurations, compare sampled and true energies, and create publication-quality plots.

4. **Custom Potentials**
   - Modify or extend potential definitions in `utils.py` to explore other energy landscapes.

## Example Outputs

- Density and energy landscape plots for the Müller-Brown potential
- Comparison between sampled energies (via dynamics) and true energies (from the potential)
- Acceptance ratio analysis for Metropolis-corrected trajectories
