"""
This script runs simulations and animates the percussion of two ice floes.
"""
import matplotlib.pyplot as plt

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

    eps = 0.75   ## coefficient de restitution

    ## Geometric constants
    X_min, X_max = 0, 50.0          # Positions of the farthest nodes in the grid
    R = 1.5 / 6.0                   # Radius for the balls (nodes) at the end of the springs

    total_length1 = 10.0
    total_length2 = 10.0
    total_length3 = 12.0

    n_nodes1 = 7
    n_nodes2 = 5
    n_nodes3 = 5

    ## Create three ice floes
    floe1 = IceFloe(nodes=None, springs=None, mass=m, stiffness=k, viscosity=mu, tenacity=L, rigid_velocity=v0,
                    id_number=0)
    floe1.generate_nodes(X_min, total_length1, n_nodes1, R)

    floe2 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=2*k_, viscosity=mu_, tenacity=3*L_, rigid_velocity=v0_,
                    id_number=1)
    floe2.generate_nodes(X_max/1.5-total_length2, X_max/1.5, n_nodes2, R)

    floe3 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=k_, viscosity=mu_, tenacity=L_, rigid_velocity=-2.0,
                    id_number=2)
    floe3.generate_nodes(X_max - total_length3, X_max, n_nodes3, R)

    f = Fracture([floe1, floe2, floe3], times=(6.0, 24.0), nStepsBefContact=500, restitutionCoef=eps)
    # f.printDetails()

    ## Run simulation and save animation
    f.runSimulation()
    f.saveFig(openFile=True, fps=10, filename="Exports/AnimFrac1D.gif")

    plt.style.use("seaborn")
    nrow, ncol = 2, 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4*(ncol+1),3.4*(nrow)))
    ax = ax.flatten()

    ## Plot positions, velocities, momentum and energy
    f.plot_positions(None, (fig, ax[0]))
    f.plot_velocities(None, (fig, ax[1]))
    f.plot_momentum((fig, ax[2]))
    f.plot_energy((fig, ax[3]))

    plt.show()

