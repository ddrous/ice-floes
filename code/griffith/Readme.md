This python package is the implementation of the numerical method for brittle fracture developed in an upcoming article.

It uses python3.

# Installation commands:
```
pip install virtualenv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Before use
### Mesh generation
Reads gmsh's .msh file in msh2 format.
We provide a couple of \*.geo file to get you started, in the mesh folder.
To generate \*.msh files from .geo files:
```
python3 mesh_generator.py 
```
or
```
gmsh -2 square_mesh.geo -o square_mesh.msh -format msh2
```

In a \*.geo file, physical group for dirichlet edges must have a 'D' in it's name, and physical group for neumann edges must have a 'N' in it's name.


### Plot options
The user can tweak the file plotrc.py to change ploting options.

# Fracture simulation : griffith_solver.py
There are some scripts available in the main directory.
The user should refer himself to the help provided in each script.
The script showcases.py is a nice introduction, as it provides a number of test cases.

### Boundary displacement
Current boundary displacements are : constant-displacement, linear-displacement, picewise-linear-displacement, rotational-displacement.
The reader is invited to read the implementation.

### Discretisation options
- Initialisation : The user can either choose a boundary-step for boundary discretisation, or fix a boundary point for fracture nucleation.
The user can also fix an interior-step for fracture initiating from the inside.
- Main discretisation : The lengh step and angle step allow the user to choose fracture precison
- Time discretisation : The user can choose between a fixed-step algorithm or the smat-time one.

### Multiprocessing
For some reason, the multiprocess option doesn't work with all versions of the module.
The version in requirements.txt works.

# Other scripts
### elastic-solver.py
Typical use :
```
python elastic_solver.py -m mesh/square.msh -cd 1
```
### broken_mesh_solver.py
Typical use :
```
python broken_mesh_solver.py 30 0 30 45 50 75 -m mesh/square.msh -cd 1
```

### admissible_fractures.py
Lists the admissible fractures for the specified discretisation options.

# TODOS
### Software enhancement
- deal with type 2 brittle fracture (i.e. add non-interpenetration on fracture lips)
- fix Software for non-convex meshes (like square_with_hole.msh)
- complete rewrite of Broken_Mesh class, with stochastic perturbation to remove exceptional cases

### Speed improvements
- rewrite the fixed_fracture solver in C (using cython ?)
- use preconditionning (scotch / parmetis ...)
- try other sparse solvers (mumps / umfpack ... )


# Other branches
### Tip Enrichement
The program supports tip enrichement according to the article of MDB99.
This enrichement is slow and is not maintained on the main branch.
An old branch (i.e. tip-enrichement) should work.

### Cython
For the slowness issue, a cython implementation has been started, and abandoned due to lack of time. 
The work done is on a cython branch.
