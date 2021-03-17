This python package is the implementation of the numerical method for the impact simulation of a mass-spring lattice of Poisson-Delaunay type.

It uses python3.

# Installation commands:
```
pip install virtualenv
python -m venv venv
source venv/bin/activate
python setup.py build_ext --inplace
pip install -r requirements.txt
```
# Use cases 
## Mass-spring lattice behavior
This is the main purpose of the code.
The simulations can be done via the cli application :
```
python springslattice-cli.py
```

Or via the web client :
```
python springslattice-web.py
```

## Linear vs nonlinear domain
The script :
```
plotlineardomain.py
```
can test if your parameters are in the linear domain or not.

## Eigengap
The script :
```
ploteigengap.py
```
plots the eigengap of the singular perturbation problem.
