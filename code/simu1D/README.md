# 1D Simulation Code

This code simulates the 1D percussion and fracture problems. Note that the fracture problem also deals with percussion.

---
## Files and Folders
- `PercussionSolver.py`: contains functions useful for the percussion problem.
- `Percussion1D-CLI.py`: define parameters and run this script to get results.
- `FractureSolver.py`: contains functions for the fracture problem.
- `Fracture1D-CLI.py`: define parameters and run this script to get results for the fracture problem (the default export filename is `Exports/AnimFrac1D.gif`).
- `requirements.txt`: Python dependencies to run the above scripts.
- `Exports/`: default output folder for simulations.
- `Notebooks/`: notebook to test specific aspects of the problems.
- `Legacy/`: old code.

---
## Instructions
In order to run these scripts:
- Create a new virtual environment and activate it (this is optional but recommended). On Linux for example:
```
$ python3 -m vnenv venv
$ source venv/bin/activate
```
- Install necessary modules: `pip3 install -r requirements.txt`
- Run the desired script e.g: `python3 Fracture1D-CLI.py` 
