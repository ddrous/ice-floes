"""
This script runs and animates the of two ice floes.
"""

import Modules.Solveur1D

if __name__=="__main__":
    ## Problem constants
    m = 1.0     ## masse du premier floe

    m_ = 1.0    ## masse du deuxième floe
    k = 15     ## raideurs
    k_ = 30

    mu = 1.9    ## viscosités
    mu_ = 1.1

    v0 = 1.8    ## vitesses avant le choc
    v_0 = 2.5

    eps = 0.4   ## coefficient de restitution



    ## Geometric constants
    L = 20.0
    H = 1.5

    L0 = L/10.0     # Spring 1 length at rest
    L_0 = L/8.0     # Spring 2 length at rest

    R = H/6.0      # Radius for the balls at the end of the springs

    z0 = 0.0
    z_0 = L-L_0


    ## Simulation times and step counts
    N=2000
    t_contact=1.0          # temps de contact

    ## Avant le contact
    tmax1 = ((z_0-R) - (z0 + L0+R)) / (v0 + v_0)       ## Calculer tmax a la main pour correspondre au moment du contact
    t = np.linspace(0, tmax1, N+1)
    t_old = t
    dt = tmax1 / N

    ## Apres le contact
    tmax2 = 4*tmax1
    N2 = 3*N
    t_new = np.linspace(0, tmax2, N2+1)


    ## Création des classes


    ## Start Flask web-interface

