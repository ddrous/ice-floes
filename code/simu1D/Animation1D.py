"""
This script runs and animates the of two ice floes.
"""

from Modules.Solveur1D import *

if __name__=="__main__":
    ## Problem constants
    m = 1.0     ## masse du premier floe

    m_ = 1.0    ## masse du deuxième floe
    k = 15     ## raideurs
    k_ = 30

    mu = 1.9    ## viscosités
    mu_ = 1.1

    v0 = 1.8    ## vitesses avant le choc
    v0_ = 2.5

    eps = 0.4   ## coefficient de restitution


    ## Geometric constants
    L = 20.0
    H = 1.5
    R = H/6.0      # Radius for the balls at the end of the springs


    ## Création des classes

    # node1 = Node([0,0], [0,0], R, 0)
    # node2 = Node([4, 0], [0,0], R, 1)
    # floe1 = IceFloe([node1, node2], id_number=0)
    # figax = floe1.plot()
    #
    # node3 = Node([10, 0], [0,0], R, 2)
    # node4 = Node([14, 0], [0,0], R, 3)
    # floe2 = IceFloe([node3, node4], id_number=1)
    # floe2.plot(figax)
    # figax[1].set_aspect('equal', adjustable='box')
    #
    # plt.show()



    ## Run a simple percussion task

    floe1 = IceFloe(nodes=None, springs=None, mass=m, stiffness=k, viscosity=mu, rigid_velocity=v0, id_number=0)
    floe1.generate_nodes(0, 4.0, 3, R)

    floe2 = IceFloe(nodes=None, springs=None, mass=m_, stiffness=k_, viscosity=mu_, rigid_velocity=v0_, id_number=1)
    floe2.generate_nodes(L-6.0, L, 4, R)

    p = Percussion(floe1, floe2, time_before_contact=4.0,
                                    time_at_contact=1.0,
                                    time_after_contact=16.0,
                                    n_steps_before_contact=2000,
                                    restitution_coef=eps)

    p.compute_before_contact()
    p.compute_at_contact()
    p.compute_after_contact()
    p.save_fig(open_file=True, filename="Animation1D.gif")


    ## Start Flask web-interface
