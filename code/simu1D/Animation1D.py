"""
This script runs and animates the of two ice floes.
"""

from Modules.Solveur1D import *


if __name__=="__main__":
    ## Problem constants
    m = 1.5     ## masse du premier floe

    m_ = 1.5    ## masse du deuxième floe
    k = 700      ## raideurs
    k_ = 800

    mu = 2.1    ## viscosités
    mu_ = 2.1

    v0 = 1.8    ## vitesses avant le choc
    v0_ = -1.8

    eps = 0.4   ## coefficient de restitution

    ## Geometric constants
    X_min, X_max = 0, 25.0        # Position of the farthest node in the grid
    R = 1.5 /6.0    # Radius for the balls at the end of the springs

    total_length1 = 5.0
    total_length2 = 8.0

    n_nodes1 = 4
    n_nodes2 = 6

    ## Run a simple percussion task

    floe1 = IceFloe(nodes=None, springs=None, mass=m, stiffness=k, viscosity=mu, rigid_velocity=v0, id_number=1)
    floe1.generate_nodes(X_min, total_length1, n_nodes1, R)

    floe2 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=k_, viscosity=mu_, rigid_velocity=v0_, id_number=2)
    floe2.generate_nodes(X_max-total_length2, X_max, n_nodes2, R)

    p = Percussion(floe1, floe2, time_before_contact=4.0,
                                    time_at_contact=0.05,
                                    time_after_contact=16.0,
                                    n_steps_before_contact=2000,
                                    restitution_coef=eps)

    p.compute_before_contact()
    # p.compute_at_contact()
    p.compute_after_contact()
    p.save_fig(open_file=True, fps=10, filename="Animation1D.gif")


    ## Start Flask web-interface


    ## Plot the energy
    plt.style.use("seaborn")
    fig, ax = plt.subplots(1, 2, figsize=(3.4*3,3.4))
    # p.plot_displacement(1, figax=(fig,ax[0]))
    p.plot_displacement(1, figax=(fig,ax[0]))
    p.plot_displacement(2, figax=(fig,ax[1]))
    plt.show()

    # print("\nDifference:", p.x2[0, 2]-p.x2[0, 1], p.x2[-1, 2]-p.x2[-1, 1])