"""
This script runs and animates the of two ice floes.
"""

from PercussionSolver import *


if __name__=="__main__":
    ## Problem constants
    m = 1.2     ## masse du premier floe
    m_ = 1.5    ## masse du deuxième floe

    k = 7      ## raideurs
    k_ = 8

    mu = 0.5    ## viscosités
    mu_ = 2.0

    L = 0.1     ## tenacities
    L_ = 2.0

    v0 = 2.2    ## vitesses avant le choc
    v0_ = -1.8

    eps = 0.4   ## coefficient de restitution

    ## Geometric constants
    X_min, X_max = 0, 40.0          # Position of the farthest node in the grid
    R = 1.5 / 6.0                   # Radius for the balls at the end of the springs

    total_length1 = 6.0
    total_length2 = 12.0

    n_nodes1 = 2
    n_nodes2 = 2

    ## Run a simple percussion task

    floe1 = IceFloe(nodes=None, springs=None, mass=m, stiffness=k, viscosity=mu, tenacity=L, rigid_velocity=v0, id_number=1)
    floe1.generate_nodes(X_min, total_length1, n_nodes1, R)

    floe2 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=k_, viscosity=mu_, tenacity=L_, rigid_velocity=v0_, id_number=2)
    floe2.generate_nodes(X_max-total_length2, X_max, n_nodes2, R)

    p = Percussion(floe1, floe2, time_before_contact=4.0,
                                    time_at_contact=0.05,
                                    time_after_contact=12.0,
                                    n_steps_before_contact=4000,
                                    restitution_coef=eps)

    ## Run the simulations and save animation
    p.run_simulation()
    # p.save_fig(open_file=True, fps=10, filename="Exports/AnimPerc1D.gif")

    ## Plot the positions
    plt.style.use("seaborn")
    fig, ax = plt.subplots(3, 2, figsize=(3.4*3,3.4*3))
    ax = ax.flatten()
    # p.plot_displacement(1, figax=(fig,ax[0]))
    p.plot_positions(1, node_ids=list(np.arange(0,6)), figax=(fig,ax[0]))
    p.plot_positions(2, node_ids=list(np.arange(0,8)), figax=(fig,ax[1]))

    ## Plot the displacements of the nodes
    p.compute_displacements()
    p.plot_displacements(1, node_ids=None, figax=(fig,ax[2]))
    p.plot_displacements(2, node_ids=None, figax=(fig,ax[3]))

    ## Study the Griffith minimization problem (for a specific ice floe)
    # p.griffith_minimization(floe_id=1)

    ## Plot the momentum and the energy of the system
    ### fig, ax = plt.subplots(1, 2, figsize=(3.4*3,3.4))
    p.plot_momentum(figax=(fig,ax[4]))
    p.plot_energy(figax=(fig,ax[5]))


    plt.show()
