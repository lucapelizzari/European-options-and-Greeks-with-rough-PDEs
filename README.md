# European-options-and-Greeks-with-rough-PDEs

## Repository structure

- `RPDE_solver.py`: finite-difference solver for the sample-wise rough PDE, including boundary-condition classes.
- `Greeks.py`: finite-difference routines for computing price, Delta, and Gamma from the RPDE solution.
- `utils_1.py`: helper functions for path interpolation, cumulative paths, plotting, and related utilities.
- `rBergomi.py`: rough Bergomi simulation utilities adapted from an external implementation.
- `rBergomi_simulation.py`: wrapper functions to generate the volatility paths, integrated driver \(I\), and quadratic variation \([I]\) used by the RPDE solver.
- `notebooks`: Currently one notebook with an illustration of code usage.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/lucapelizzari/European-options-and-Greeks-with-rough-PDEs.git
cd rough-pde-pricing
pip install -e ".[dev]"
