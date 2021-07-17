"""
This script runs and animates the of two ice floes.
"""

from FractureSolver import *


if __name__=="__main__":
    ## Problem constants
    m = 1.2     ## masse du premier floe

    m_ = 1.5    ## masse du deuxième floe
    k = 70      ## raideurs
    k_ = 80

    mu = 0.5    ## viscosités
    mu_ = 2.0

    L = 0.1     ## tenacities
    L_ = 2.0

    v0 = 2.2    ## vitesses avant le choc
    v0_ = -1.8

    eps = 0.4   ## coefficient de restitution

    ## Geometric constants
    X_min, X_max = 0, 40.0        # Position of the farthest node in the grid
    R = 1.5 / 6.0    # Radius for the balls at the end of the springs

    total_length1 = 6.0
    total_length2 = 12.0

    n_nodes1 = 4
    n_nodes2 = 3

    ## Run a simple percussion task

    floe1 = IceFloe(nodes=None, springs=None, mass=m, stiffness=k, viscosity=mu, tenacity=L, rigid_velocity=v0,
                    id_number=0)
    floe1.generate_nodes(X_min, total_length1, n_nodes1, R)

    floe2 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=k_, viscosity=mu_, tenacity=L_, rigid_velocity=v0_,
                    id_number=1)
    floe2.generate_nodes(X_max-total_length2, X_max, n_nodes2, R)

    ## Create two ice floes
    f = Fracture([floe1, floe2, floe2], times=(4.0,12.0), nStepsBefContact=4000, restitutionCoef=eps)
    f.printDetails()


