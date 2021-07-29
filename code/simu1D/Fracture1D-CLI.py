"""
This script runs and animates the of two ice floes.
"""

from FractureSolver import *


if __name__=="__main__":
    ## Problem constants
    m = 1.2     ## masse du premier floe
    m_ = 1.5    ## masse du deuxième floe

    k = 7.0      ## raideurs
    k_ = 8.0

    mu = 0.5    ## viscosités
    mu_ = 2.0

    L = 0.1     ## tenacities
    L_ = 6.0

    v0 = 2.2    ## vitesses avant le choc
    v0_ = -1.8

    eps = 0.95   ## coefficient de restitution

    ## Geometric constants
    X_min, X_max = 0, 50.0        # Position of the farthest node in the grid
    R = 1.5 / 6.0    # Radius for the balls at the end of the springs

    total_length1 = 10.0
    total_length2 = 10.0
    total_length3 = 12.0

    n_nodes1 = 7
    n_nodes2 = 5
    n_nodes3 = 5

    ## Run a simple percussion task

    floe1 = IceFloe(nodes=None, springs=None, mass=m, stiffness=k, viscosity=mu, tenacity=L, rigid_velocity=v0,
                    id_number=0)
    floe1.generate_nodes(X_min, total_length1, n_nodes1, R)

    floe2 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=2*k_, viscosity=mu_, tenacity=3*L_, rigid_velocity=v0_,
                    id_number=1)
    floe2.generate_nodes(X_max/1.3-total_length2, X_max/1.5, n_nodes2, R)

    floe3 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=k_, viscosity=mu_, tenacity=L_, rigid_velocity=-2.0,
                    id_number=2)
    floe3.generate_nodes(X_max - total_length3, X_max, n_nodes3, R)

    ## Create two ice floes
    f = Fracture([floe1, floe2, floe3], times=(6.0, 24.0), nStepsBefContact=500, restitutionCoef=eps)
    # f.printDetails()
    # f.computeBeforeContact()
    # f.computeAfterContact()
    f.runSimulation()
    f.saveFig(openFile=True, fps=5, filename="Exports/AnimFrac1D.gif")

    # print("CONFIG", f.configurations)


