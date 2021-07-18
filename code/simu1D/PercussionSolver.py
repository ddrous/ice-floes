"""
This module defines classes and functions for the collision and displacement of ice floes along with their 2 nodes.
"""

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import imageio, io
import os
import PIL.Image as PILImage

from itertools import *

import sys
sys.setrecursionlimit(10**4)      # Recursion is important for this problem


""" 
Four classes: "Node -> Springs -> IceFloe -> Percussion". 
A percussion problem is a collision problem between 
two ice floes. Each ice floe is composed of several
nodes and springs. Each spring is built from two nodes. 
"""



class Node:
    """
    A class representing one node of an ice floe
    """
    def __init__(self, position, velocity, radius, id_number):
        self.x, self.y = position
        self.x0, self.y0 = position     ## Initial position needed for plots
        self.vx, self.vy = velocity
        self.R = radius
        self.id = id_number
        ## Extra properties needed for fracture
        self.parentFloe = None     ## Ice floe to which this node belongs
        self.leftNode = None
        self.rightNode = None
        self.leftSpring = None
        self.rightSpring = None

    def __str__(self):
        return "Node ["+ str(self.id) +"] properties: \n" \
               + "  position: " + str(self.x) + "\n" \
               + "  velocity: " + str(self.vx) + "\n" \
               + "  radius: " + str(self.R) + "\n" \
               + "  parent floe: " + str(self.parentFloe) + "\n" \
               + "  neighbouring nodes: (" + str(self.leftNode) + ", " + str(self.rightNode) + ")\n" \
               + "  neighbouring springs: (" + str(self.leftSpring) + ", " + str(self.rightSpring) + ")\n"

    def plot(self, figax=None):
        fig, ax = figax

        c = Circle((self.x, self.y), self.R, fc='white', ec='k', lw=2, zorder=10)
        ax.add_patch(c)
        return figax




class Spring:
    """
    A class representing one spring of an ice floe
    """
    def __init__(self, node1:Node, node2:Node, diameter, id_number):
        self.node1 = node1
        self.node2 = node2
        self.L0 = d_nodes(node1, node2)
        self.D = diameter
        self.theta = np.arctan2(node2.y-node1.y, node2.x-node1.x) + np.pi/2.0
        self.id = id_number
        ## Extra parameters for fracture
        self.parentFloe = None  ## Ice floe to which this spring belongs
        self.leftNode = None
        self.rightNode = None

    def plot(self, figax=None):
        """
        Plot the spring from node1 to node2 as the projection of a helix.
        """
        fig, ax = figax

        # Spring turn radius, number of turns
        rs, ns = 0.05, int(5 * self.L0)

        # Number of data points for the helix
        Ns = 500

        # Length of the spring
        startx = self.node1.x + self.node1.R
        starty = self.node1.y
        if self.node1.x > self.node2.x:
            startx = self.node2.x
            starty = self.node2.y
        elif self.node1.x == self.node2.x:
            print("Carefull: zero sized spring!")
        L = d_nodes(self.node2, self.node1) - self.node1.R - self.node2.R
        # assert L > 0, "Impossible: Negative length spring"
        # if L < 0: print("Carefull: A spring has a negative length")

        # We don't draw coils all the way to the end of the spring: we pad a bit
        ipad1, ipad2 = 50, 50
        w = np.linspace(0, L, Ns)

        # Set up the helix along the x-axis
        xp = np.zeros(Ns)
        xp[ipad1:-ipad2] = rs * np.sin(2 * np.pi * ns * w[ipad1:-ipad2] / L)

        # Then rotate it to align with any desired axis (x-axis)
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta), np.cos(self.theta)]])
        xs, ys = - R @ np.vstack((xp, w))

        ax.plot(startx + xs, starty + (0.9*self.D * ys) / np.max(ys), c='k', lw=2)

        return figax




class IceFloe:
    """
    A class representing an ice floe
    """
    def __init__(self, nodes=None, springs=None, mass=1.0, stiffness=15, viscosity=2.0, tenacity=1.0, rigid_velocity=None, id_number=None):
        if nodes:
            self.nodes = nodes
            self.n = len(nodes)
        else:
            print("Ice floe "+str(id_number)+" created but contains no nodes")

        if springs:
            self.springs = springs
        else:
            print("Ice floe "+str(id_number)+" created but contains no springs")

        self.m = mass
        self.k = stiffness
        self.mu = viscosity
        self.L = tenacity
        self.v0 = rigid_velocity       ## One velocity for all nodes
        self.id = id_number

    def generate_nodes(self, start, end, count, radius):
        """
        Builds regularly spaced nodes and adds them to the floe
        """
        assert self.v0 is not None, "Cannot determine velocity for new nodes"

        self.nodes = []
        # for i, x in enumerate(np.arange(start, end, (start-end)/count)):
        #     self.nodes.append(Node([x,0], [self.v0, 0], radius, i))

        x = np.linspace(start, end, count)
        for i in range(count):
            self.nodes.append(Node([x[i],0], [self.v0, 0], radius, i))

        self.n = len(self.nodes)
        print("Generated nodes for ice floe "+str(self.id))

        ## Also generate the springs in here if you like
        self.generate_springs()
        self.init_lengths = self.initial_lengths()      ## Array of all spring lengths


    def generate_springs(self):
        """
        Builds springs and adds them to the floe
        """
        assert self.nodes is not None, "Cannot build springs if nodes are absent."

        self.springs = []
        for i in range(self.n - 1):
            spring = Spring(self.nodes[i],
                            self.nodes[i + 1],
                            (self.nodes[i].R + self.nodes[i + 1].R) / 2.0,
                            i)
            self.springs.append(spring)

        print("Generated springs for ice floe "+str(self.id))

    def velocities_array(self):
        """
        Returns velocities of all nodes along x axis
        """
        v = np.zeros((self.n))
        for i, node in enumerate(self.nodes):
            v[i] = node.vx

        return v

    def positions_array(self):
        """
        Returns positions of all nodes from their equilibrium (nodes along x axis)
        """
        x = np.zeros((self.n))
        for i, node in enumerate(self.nodes):
            x[i] = node.x

        return x

    def initial_lengths(self):
        """
        Initial lengths of springs
        """
        L0 = np.zeros((self.n-1))
        for i, spring in enumerate(self.springs):
            L0[i] = spring.L0

        return L0


    def max_radius(self):
        """
        Returns the radius of the biggest node
        """
        r = np.zeros((self.n))
        for i, node in enumerate(self.nodes):
            r[i] = node.R
        return np.max(r)

    def update_along_x(self, x_array, vx_array):
        """
        Update the position and speeds of all the nodes in the ice floe
        """
        for i, node in enumerate(self.nodes):
            node.x = x_array[i]
            node.vx = vx_array[i]

    def plot(self, figax=None):
        """
        Plot an ice floe whose nodes are at (x1,y1) and (x2,y2) with same radius R
        """
        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        for node in self.nodes:
            node.plot(figax)

        for spring in self.springs:
            spring.plot(figax)

        return figax




class Percussion:
    """
    A class representing a percussion problem between two floes
    __Note__: Normally, z denotes the position and x the displacement.
    However, in the code below, `x1` and `x2` represent the positions
    of ice floes 1 and 2 respectively. The displacements are later
    references are `dep1` and `dep2` respectively.
    """
    def __init__(self, floe1:IceFloe, floe2:IceFloe, time_before_contact=4.0, time_at_contact=1.0, time_after_contact=16.0, n_steps_before_contact=2000, restitution_coef=0.4):
        self.floe1, self.floe2 = floe1, floe2
        self.eps = restitution_coef

        self.t_bef, self.t_at, self.t_aft = time_before_contact, time_at_contact, time_after_contact
        self.N_bef = n_steps_before_contact
        self.N_at = n_steps_before_contact//10
        self.N_aft = int(n_steps_before_contact*(time_after_contact/time_before_contact))

        self.rec_count = 0      ## Recursion depth counter (1 recurson for 1 collision)
        self.contact_indices = []

        ## Initial positions for the arrays
        self.init_pos1 = floe1.positions_array()
        self.init_pos2 = floe2.positions_array()

        ## Index at which fracture occcurs in each ice floes
        self.fracture_indices1 = []
        self.fracture_indices2 = []

    def compute_before_contact(self):
        self.t = np.linspace(0, self.t_bef, self.N_bef+1)
        self.x1 = np.zeros((self.t.size, self.floe1.n))      ## Positions for floe1
        self.x2 = np.zeros((self.t.size, self.floe2.n))      ## Positions for floe2

        self.v1 = self.floe1.v0 * np.ones((self.t.size, self.floe1.n))      ## Velocities along x for floe1
        self.v2 = self.floe2.v0 * np.ones((self.t.size, self.floe2.n))      ## Velocities along x for floe2

        self.x1[:, 0] = simulate_uniform_mov(self.floe1.nodes[0].x, self.floe1.v0, self.t)
        self.x2[:, 0] = simulate_uniform_mov(self.floe2.nodes[0].x, self.floe2.v0, self.t)

        for i in range(1, self.floe1.n):
            self.x1[:,i] = self.x1[:,0] + (self.floe1.nodes[i].x - self.floe1.nodes[0].x)

        for i in range(1, self.floe2.n):
            self.x2[:,i] = self.x2[:,0] + (self.floe2.nodes[i].x - self.floe2.nodes[0].x)

        ## Check whether IceFloe1 and IceFloe2 will collide
        collided = self.check_colission()
        if not collided:
            ## Double simulation time and run phase 1 again
            self.t_bef = self.t_bef * 2
            self.t_aft = self.t_aft * 2
            self.compute_before_contact()
        else:
            ## Then phase 1 is OK
            return

    def compute_at_contact(self):
        """
        Computes the resulting velocities of the two colliding nodes
        """

        ## Compute the integrand for speed calculation
        # t_con, xvx1 = simulate_displacement_wrapper(self.floe1, self.t_at, self.N_at)
        # t_con, xvx2 = simulate_displacement_wrapper(self.floe2, self.t_at, self.N_at)

        # intgr = self.floe1.k*(xvx1[:,self.floe1.n-2] - xvx1[:,self.floe1.n-1] + self.floe1.springs[-1].L0) \
        #         + self.floe1.mu*(xvx1[:,-2] - xvx1[:,-1]) \
        #         - self.floe2.k*(xvx2[:,0] - xvx2[:,1] + self.floe2.springs[0].L0) \
        #         - self.floe2.mu*(xvx2[:,self.floe2.n] - xvx2[:, self.floe2.n+1])

        # I = np.trapz(y=intgr, x=t_con)
        # ## I = 0       ## Conservation de la quantité de mouvement
        # print("Value of I for computation:", I)

        ## Compute the velocities after contact
        v0 = np.abs(self.floe1.nodes[-1].vx)
        v0_ = np.abs(self.floe2.nodes[0].vx)
        # v0 = self.floe1.nodes[-1].vx
        # v0_ = self.floe2.nodes[0].vx
        m = self.floe1.m
        m_ = self.floe2.m
        eps = self.eps

        ##------------Solution par défaut----------------------------------
        #    # V0 = (I + (m - eps * m_) * v0 + (1 + eps) * m * v0_) / (m + m_)
        #    # V0_ = (I + (1 + eps) * m * v0 + (m_ - eps * m) * v0_) / (m + m_)

        # V0 = (I + (m + eps * m_) * v0 + (1.0 - eps) * m * v0_) / (m + m_)
        # V0_ = (I + (1.0 - eps) * m * v0 + (m_ + eps * m) * v0_) / (m + m_)
        ##-------------------------------------------------------------

        ##------------1er alternative----------------------------------
        # v0 = self.floe1.nodes[-1].vx
        # v0_ = self.floe2.nodes[0].vx
        #
        # ## Case 1
        # A1 = np.array([[m, m_], [1.0, -1.0]])
        # # b1 = np.array([m*np.abs(v0) + m_*np.abs(v0_), eps*np.abs(v0 - v0_)])
        # b1 = np.array([m*np.abs(v0) + m_*np.abs(v0_), eps*(v0 - v0_)])
        # V0, V0_ = list(np.linalg.inv(A1) @ b1)
        # # print("MATRIX", A1)
        # # print("INVERSE", np.linalg.inv(A1))
        #
        # ## Case 2
        # A2 = np.array([[m, m_], [-1.0, 1.0]])
        # V0, V0_ = list(np.linalg.inv(A2) @ b1)
        #
        # test1 =  (A1 @ np.array([V0, V0_]))[-1] > 0.0
        # test2 =  (A2 @ np.array([V0, V0_]))[-1] > 0.0
        # print("Testing positions", test1, test2)
        ##-------------------------------------------------------------

        ##------------2eme alternative----------------------------------
        # X = m*v0 + m_*v0_
        # Y = m*(v0**2) + m_*(v0_**2)
        # a = (m**2/m_ + m)
        # b = -2*X*m/m_
        # c = (X**2*m_) - Y
        # Delta = b**2 - 4*a*c
        # V01 = (-b - np.sqrt(Delta)) / (2*a)
        # V01 = (-b + np.sqrt(Delta)) / (2*a)
        # print("V0 values:", Delta, V01, V01)
        ##-------------------------------------------------------------

        ##------------3eme alternative----------------------------------
        X = eps*(v0 - v0_)
        Y = m*(v0**2) + m_*(v0_**2)
        a = m+m_
        b = 2*m_*X
        c = m_*(X**2) - Y
        Delta = b**2 - 4*a*c
        V01 = (-b - np.sqrt(Delta)) / (2*a)
        V02 = (-b + np.sqrt(Delta)) / (2*a)
        # print("V0 values:", Delta, V01, V02)
        V0 = V01 if V01 >= 0 else V02
        V0_ = V0 + X
        # print("TEST:", m*(V0**2) + m_*(V0_**2) == Y)
        ##-------------------------------------------------------------

        print("VELOCITIES BEFORE/AFTER CONTACT:")
        print(" First floe:", [v0, -np.abs(V0)])
        print(" Second floe:", [-v0_, np.abs(V0_)])
        # print(" First floe:", [v0, -V0])
        # print(" Second floe:", [v0_, V0_])

        ## Update velocities at extreme nodes
        self.floe1.nodes[-1].vx = -np.abs(V0)
        self.floe2.nodes[0].vx = np.abs(V0_)
        # self.floe1.nodes[-1].vx = -V0
        # self.floe2.nodes[0].vx = V0_


    def compute_after_contact(self):
        """
        Computes the positions and velocities of the two colliding floes after a contact
        """
        self.compute_at_contact()       ## Calculate new speeds ...

        t_sim, xvx1 = simulate_displacement_wrapper(self.floe1, self.t_aft, self.N_aft)
        t_sim, xvx2 = simulate_displacement_wrapper(self.floe2, self.t_aft, self.N_aft)

        # print(xvx1[:, -1])

        # t_sim, xvx1 = simulate_displacement_wrapper(self.floe1, "left", self.t_aft, self.N_aft)
        # t_sim, xvx2 = simulate_displacement_wrapper(self.floe2, "right", self.t_aft, self.N_aft)

        self.t = np.concatenate([self.t, self.t[-1] + t_sim])

        # dx1 = self.floe1.displacements_array(-1)
        self.x1 = np.concatenate([self.x1, xvx1[:, :self.floe1.n]])
        self.v1 = np.concatenate([self.v1, xvx1[:, self.floe1.n:]])

        # dx2 = self.floe2.displacements_array(0)
        self.x2 = np.concatenate([self.x2, xvx2[:, :self.floe2.n]])
        self.v2 = np.concatenate([self.v2, xvx2[:, self.floe2.n:]])

        print("Recursion depth:", self.rec_count)

        ## Check collision then recalculate if applicable
        collided = self.check_colission(self.contact_indices[-1] + 2)
        if (not collided) or (self.t.size > self.N_bef+self.N_aft) or (self.rec_count > 9800):
                return
        else:
            self.rec_count += 1
            self.compute_after_contact()


    def check_colission(self, start_index=0):
        """
        Checks is the two floes will collide. If that is the case, save each nodes'
        position and velocity, then discard the remainder of the tensors.
        """
        assert start_index < self.t.size, "Starting index to check collision too big"

        collided = False
        for i in range(start_index, self.t.size):
            if self.x1[i,-1]+self.floe1.nodes[-1].R > self.x2[i,0]-self.floe2.nodes[0].R:

                ## If collision, save each nodes positions and speed
                for j, node in enumerate(self.floe1.nodes):
                    node.x = self.x1[i,j]
                    node.vx = self.v1[i,j]

                for j, node in enumerate(self.floe2.nodes):
                    node.x = self.x2[i,j]
                    node.vx = self.v2[i,j]

                collided = True
                self.contact_indices.append(i)
                break

        if collided:
            ## Discard the positions and velocities after collision
            self.x1 = self.x1[:self.contact_indices[-1] + 1]
            self.v1 = self.v1[:self.contact_indices[-1] + 1]
            self.x2 = self.x2[:self.contact_indices[-1] + 1]
            self.v2 = self.v2[:self.contact_indices[-1] + 1]
            self.t = self.t[:self.contact_indices[-1] + 1]

        return collided

    def plot_positions(self, floe_id:int, node_ids=None, figax=None):
        """
        Plot the positions of (some) nodes of an ice floe part of this percussion problem.
        """
        if floe_id == self.floe1.id:
            floe = self.floe1
            x = self.x1
        elif floe_id == self.floe2.id:
            floe = self.floe2
            x = self.x2
        else:
            print("Ice floe of id "+str(floe_id)+" is not part of this problem.")
            return figax

        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        if node_ids==None:
            node_ids = np.arange(floe.n)
        try:
            for i in node_ids:
                ax.plot(self.t, x[:, i], label=r"$z_"+str(i)+"$")
        except IndexError:
            print("Error plotting: A given node id not valid!")

        ax.set_title("Trajectoires des noeuds du floe "+str(floe_id))
        ax.set_xlabel("temps")
        ax.legend()
        fig.tight_layout()

        return (fig, ax)

    def compute_displacements(self):
        """
        This function computes the displacement of each node in the percussion
        by eliminating the movement of the ice floes as a whole.
        """

        self.gravity_pos1 = np.mean(self.x1, axis=1)
        self.gravity_pos2 = np.mean(self.x2, axis=1)

        gravity_init_pos1 = np.mean(self.init_pos1)
        gravity_init_pos2 = np.mean(self.init_pos2)

        self.dep1 = (self.x1 - self.gravity_pos1[:, np.newaxis]) - (self.init_pos1 - gravity_init_pos1)
        self.dep2 = (self.x2 - self.gravity_pos2[:, np.newaxis]) - (self.init_pos2 - gravity_init_pos2)

    def plot_displacements(self, floe_id:int, node_ids=None, figax=None):
        """
        Plots the displacement of each node following their computation in compute_displacements().
        """
        if floe_id == self.floe1.id:
            floe = self.floe1
            dep = self.dep1
        elif floe_id == self.floe2.id:
            floe = self.floe2
            dep = self.dep2
        else:
            print("Ice floe of id " + str(floe_id) + " is not part of this problem.")
            return figax

        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        if node_ids==None:
            node_ids = np.arange(floe.n)
        try:
            for i in node_ids:
                ax.plot(self.t, dep[:, i], label=r"$x_"+str(i)+"$")
        except IndexError:
            print("Error plotting: A given node id not valid!")

        ax.set_title("Déplacements des noeuds du floe "+str(floe_id))
        ax.set_xlabel("temps")
        ax.legend()
        fig.tight_layout()

        return (fig, ax)

    def save_fig(self, fps=24, filename="Animation1D.gif", open_file=True):
        """
        Plot both ice floes whose nodes are at (x1,y1) and (x2,y2) with same radius R
        """
        min_X = self.floe1.nodes[0].x0 - self.floe1.nodes[0].R
        max_X = self.floe2.nodes[-1].x0 + self.floe2.nodes[-1].R
        max_R = np.max([self.floe1.max_radius(), self.floe2.max_radius()])

        plt.style.use("default")
        fig = plt.figure(figsize=(max_X-min_X, 5*max_R), dpi=72)
        ax = fig.add_subplot(111)

        # ax.set_xlim(min_X, max_X)
        # ax.set_ylim(-4 * max_R, 4 * max_R)
        # ax.set_aspect('equal', adjustable='box')

        dt = self.t_bef/self.N_bef
        di = int(1 / fps / dt)

        img_list = []

        print("Generating frames ...")
        ## For loop to update the floes nodes, then plot
        for i in range(0, self.t.size, di):
            print("  ", i // di, '/', self.t.size // di)

            self.floe1.update_along_x(self.x1[i,:], self.v1[i,:])
            self.floe2.update_along_x(self.x2[i,:], self.v2[i,:])

            self.floe1.plot(figax=(fig,ax))
            self.floe2.plot(figax=(fig,ax))

            ax.set_xlim(min_X, max_X)
            ax.set_ylim(-2 * max_R, 2 * max_R)
            ax.set_aspect('equal', adjustable='box')

            img_list.append(fig2img(fig))

            plt.cla()      # Clear the Axes ready for the next image.

        imageio.mimwrite(filename, img_list)
        print("OK! saved file '"+filename+"'")

        if open_file:
            ## Open animation
            os.system('gthumb '+filename)     ## Only on Linux

    def plot_momentum(self, figax):
        """
        Plots the momentum of the system before and after first choc
        """
        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        P_av = (self.floe1.n * self.floe1.m * np.abs(self.floe1.v0)
                + self.floe2.n * self.floe2.m * np.abs(self.floe2.v0)) * np.ones_like(self.t)
        # P_av = self.floe1.m * np.sum(np.abs(self.v1), axis=-1) + self.floe2.m * np.sum(np.abs(self.v2), axis=-1)
        N_first = self.contact_indices[0]
        P_av[N_first + 1:] = np.nan

        P_ap = self.floe1.m * np.sum(np.abs(self.v1), axis=-1) + self.floe2.m * np.sum(np.abs(self.v2), axis=-1)
        P_ap[:N_first + 1] = np.nan

        print("\nQuantité de mouvement immediatement avant 1er choc:", P_av[N_first])
        print("Quantité de mouvement immediatement après 1er choc:", P_ap[N_first+1])
        # print("Rapport APRÈS/AVANT:", P_ap[N_first+1] / P_av[N_first])
        # print("Epsilon:", self.eps)

        ax.plot(self.t, P_av, label="avant 1er choc")
        ax.plot(self.t, P_ap, label="après 1er choc")
        for i, N_choc in enumerate(self.contact_indices[:]):
            label = "chocs" if i==0 else None
            ax.plot([self.t[N_choc+1]], [P_ap[N_choc+1]], 'kX', alpha=0.5, label=label)

        ax.set_title("Quantité de mouvement")
        ax.set_xlabel("temps")
        text = 'rapport fin/début: ' + str(np.round(P_ap[-1] / P_av[0], 2)) \
               + '\nepsilon: ' + str(self.eps)
        ax.text(0.9, 0.1, text,
             horizontalalignment='right',
             verticalalignment='baseline',
             transform=ax.transAxes)
        ax.legend()
        fig.tight_layout()

        return (fig, ax)

    def plot_energy(self, figax):
        """
        Plots the total energy of the system before and after first choc
        """
        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        N_first = self.contact_indices[0]

        ## Energie avant choc
        E_av = (self.floe1.n * 0.5 * self.floe1.m * self.floe1.v0**2
                + self.floe2.n * 0.5 * self.floe2.m * self.floe2.v0**2) * np.ones_like(self.t)
        E_av[N_first + 1:] = np.nan

        ## Energie cinetique apres choc
        E_ap_c = (0.5 * self.floe1.m * np.sum(self.v1**2, axis=-1) \
                 + 0.5 * self.floe2.m * np.sum(self.v2**2, axis=-1)) * np.ones_like(self.t)

        ## Energie elastique apres choc
        E_ap_el = 0.5 * self.floe1.k * np.sum((self.x1[:, 1:] - self.x1[:, :-1] - self.floe1.initial_lengths())**2, axis=-1) \
                    + 0.5 * self.floe2.k * np.sum((self.x2[:, 1:] - self.x2[:, :-1] - self.floe2.initial_lengths())**2, axis=-1)

        ## Energie dissipative apres choc
        # unit1 = (self.x1[:, 1:] - self.x1[:, :-1]) / np.linalg.norm((self.x1[:, 1:] - self.x1[:, :-1]))
        # unit2 = (self.x2[:, 1:] - self.x2[:, :-1]) / np.linalg.norm((self.x2[:, 1:] - self.x2[:, :-1]))
        # E_ap_r_OLD = 0.5 * self.floe1.mu * np.sum(((self.v1[:, 1:] - self.v1[:, :-1]) * unit1)**2, axis=-1) \
        #         + 0.5 * self.floe2.mu * np.sum(((self.v2[:, 1:] - self.v2[:, :-1]) * unit2)**2, axis=-1)
        # # E_ap_r_OLD = 0.5 * self.floe1.mu * np.sum(((self.v1[:, 1:] - self.v1[:, :-1]))**2, axis=-1) \
        # #         + 0.5 * self.floe2.mu * np.sum(((self.v2[:, 1:] - self.v2[:, :-1]))**2, axis=-1)
        # # E_ap_r_OLD = 0.5 * self.floe1.mu * np.sum((np.abs(self.v1[:, 1:]) - np.abs(self.v1[:, :-1])) ** 2, axis=-1) \
        # #          + 0.5 * self.floe2.mu * np.sum((np.abs(self.v2[:, 1:]) - np.abs(self.v2[:, :-1])) ** 2, axis=-1)
        E_ap_r_OLD = self.floe1.mu * np.sum((self.v1[:, 1:] - self.v1[:, :-1]) ** 2, axis=-1) \
                    + self.floe2.mu * np.sum((self.v2[:, 1:] - self.v2[:, :-1]) ** 2, axis=-1)
        integrand = E_ap_r_OLD[N_first:]
        t = self.t[N_first:] - self.t[N_first-1:-1]
        E_ap_r = np.zeros_like(E_ap_el)
        E_ap_r[N_first:] = np.cumsum(integrand*t)

        E_ap = E_ap_c + E_ap_el + E_ap_r
        # E_ap = E_ap_r
        E_ap[:N_first + 1] = np.nan
        E_ap_c[:N_first + 1] = np.nan
        E_ap_el[:N_first + 1] = np.nan
        E_ap_r[:N_first + 1] = np.nan

        print("Énergie totale immediatement avant 1er choc:", E_av[N_first])
        print("Énergie totale immediatement après 1er choc:", E_ap[N_first + 1])
        # print("Rapport APRÈS/AVANT:", E_ap[N_first + 1] / E_av[N_first])
        # print("Epsilon:", self.eps)

        ax.plot(self.t, E_av, label="énergie totale avant 1er choc")
        ax.plot(self.t, E_ap, label="énergie totale après 1er choc")
        ax.plot(self.t, E_ap_c, "--", linewidth=1, label="énergie cinétique")
        ax.plot(self.t, E_ap_el, "--", linewidth=1, label="énergie élastique")
        ax.plot(self.t, E_ap_r, "--", linewidth=1, label="énergie dissipée")

        for i, N_choc in enumerate(self.contact_indices):
            # label = "1er" if i==0 else str(i+1)+"eme"
            # ax.plot([self.t[N_choc+1]], [E_ap[N_choc+1]], marker='X', label=label+" choc")
            label = "chocs" if i==0 else None
            ax.plot([self.t[N_choc+1]], [E_ap[N_choc+1]], 'kX', alpha=0.5, label=label)

        ##----------- Bonus:Plot Fracture Point ------------##
        if self.fracture_indices1:
            frac = self.fracture_indices1[-1]
            ax.plot([self.t[frac+1]], [E_ap_el[frac+1]], 'rX', label="fracture")
        ##--------------------------------------------------##

        ax.set_title("Énergie totale")
        ax.set_xlabel("temps")
        text = 'rapport fin/début: ' + str(np.round(E_ap[-1] / E_av[0], 2)) \
               + '\nepsilon: ' + str(self.eps)
        ax.text(0.9, 0.1, text,
             horizontalalignment='right',
             verticalalignment='baseline',
             transform=ax.transAxes)
        ax.legend()
        fig.tight_layout()

        return figax

    def fracture_energy(self, floe_id=None, broken_springs=None):
        """
        Computes the fracture energy of a fractured ice floe, i.e. some springs and broken.
        """
        if floe_id == self.floe1.id:
            floe = self.floe1
        elif floe_id == self.floe2.id:
            floe = self.floe2
        else:
            print("Ice floe of id " + str(floe_id) + " is not part of this problem.")

        ### Bien préciser qu'on est en déformation élastqiue: et donc la longueur de la fracture est la longeur initiale des ressorts
        broken_length = 0
        for i in broken_springs:
            try:
                broken_length += floe.springs[i].L0
            except IndexError:
                print("Error: Spring id "+str(i)+" is not a valid id for ice floe "+str(floe_id))

        return floe.L * broken_length

    def deformation_energy(self, floe_id=None, broken_springs=None, start=0, end=0):
        """
        Computes the deformation energy (sum of the elastic energy and the dissipated
        energy) when the ice floe is fractured, i.e. some springs and broken.
        """
        assert start <= end, "Error: Starting time step is bigger than ending time step!"

        if floe_id == self.floe1.id:
            floe = self.floe1
            x, v = self.x1, self.v1
        elif floe_id == self.floe2.id:
            floe = self.floe2
            x, v = self.x2, self.v2
        else:
            print("Ice floe of id " + str(floe_id) + " is not part of this problem.")

        ## Energie potentielle elastique (au pas de temps `end`)
        k = np.full((floe.n - 1), floe.k)
        k[broken_springs] = 0.0         ## Eliminate the broken springs

        E_el = 0.5 * np.sum(k * (x[end, 1:] - x[end, :-1] - floe.init_lengths)**2, axis=-1)

        ## Energie dissipée entre les temps `start` et `end`
        mu = np.full((floe.n - 1), floe.mu)
        mu[broken_springs] = 0.0        ## Eliminate the broken dampers

        E_ap_r_OLD = np.sum(mu * (v[:end+1, 1:] - v[:end+1, :-1])**2, axis=-1)
        integrand = E_ap_r_OLD[start:]
        t = self.t[start:end+1] - self.t[start-1:end]
        E_r = np.sum(integrand*t)

        # return E_el + E_r     #### ---- STUDY THIS PART AGAIN ---- ####
        return E_el


    def griffith_minimization(self, floe_id=None):
        """
        Studies the fracture problem to see if it is worth addind a path tot the crack.
        __Note__: Here, the total energy is not the same as in `self.plot_energy()`. Here,
        it is the sum of the deformation energy and the fracture energy.
        """
        if floe_id == self.floe1.id:
            floe = self.floe1
            fracture_indices = self.fracture_indices1
        elif floe_id == self.floe2.id:
            floe = self.floe2
            fracture_indices = self.fracture_indices2
        else:
            print("Ice floe of id " + str(floe_id) + " is not part of this problem.")

        print("\nGRIFFITH FRACTURE STUDY FOR ICE FLOE " + str(floe_id) + ':')

        ## Specify the time steps for the computations
        start = self.contact_indices[-1]
        old_end = start                  ## Time step for current energy

        ## Compute the current energy of the system (no broken spring)
        old_broken_springs = []
        def_en = self.deformation_energy(floe_id, old_broken_springs, start, old_end)
        frac_en = self.fracture_energy(floe_id, old_broken_springs)
        old_energy = def_en + frac_en
        # print("OLD ENERGY IS:", old_energy)

        ## Compute new energies, only stop is fracture of end of simulation
        steps_counter = 0
        while (True):
            steps_counter += 10
            new_end = old_end + steps_counter  ## Time step for new energy and crack path
            energies = {}
            ## Identifies all possible path cracks (easy in 1D)
            ## Identifies all possible displacements (do we restart the simulation?)
            #### (For now, we use the one issued form the percussion)
            ## Computes all possible total energies for new paths and new displacements
            for i in range(1, floe.n):
                for new_tuple in combinations(range(floe.n-1), i):
                    new_broken_springs = list(new_tuple)
                    def_en = self.deformation_energy(floe_id, new_broken_springs, start, new_end)
                    frac_en = self.fracture_energy(floe_id, new_broken_springs)
                    energies[new_tuple] = def_en+frac_en

            min_config = sorted(energies.items(), key=lambda item: item[1])[0]

            ## Compare to the old energy and conclude
            if min_config[1] < old_energy:
                print("     Starting configuration was:", (tuple(old_broken_springs), old_energy))
                # print("     New Griffith competing energies is:", energies)
                print("     Minimum energy reached for:", min_config)
                # print("     Is there a Fracture?:", min_config[1] < old_energy)
                print("     Fracture happens", steps_counter, "time step(s) after last collision, at time:", self.t[new_end])
                fracture_indices.append(new_end)
                break
            if new_end >= self.N_bef+self.N_aft:
                print("     During the whole simulation, there was no fracture !")
                break







""" General purpose functions. The function simulate_displacement is the most heart of the problem """

def d_nodes(node_1, node_2):
    """
    Distance between two nodes
    """
    return np.sqrt((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2)

def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image and return it
    """
    buf = io.BytesIO()
    # fig.savefig(buf, bbox_inches='tight')
    fig.savefig(buf)
    buf.seek(0)
    img = PILImage.open(buf)
    return img

def simulate_displacement(n=2, m=1.0, k=18.0, mu=1.3, x0=None, v0=None, L0=None, t_simu=1.0, N=1000):
    """
    Calculates the positions and velocities of an ice floe as a dynamical system
    """
    diagB = -2.0 * k * np.ones((n))
    diagB[0] = -k
    diagB[-1] = -k
    B = np.diag(diagB / m) + np.diag(k * np.ones((n - 1)) / m, 1) + np.diag(k * np.ones((n - 1)) / m, -1)

    diagC = -2.0 * mu * np.ones((n))
    diagC[0] = -mu
    diagC[-1] = -mu
    C = np.diag(diagC / m) + np.diag(mu * np.ones((n - 1)) / m, 1) + np.diag(mu * np.ones((n - 1)) / m, -1)

    E = np.zeros((2 * n, 2 * n))
    E[:n, n:] = np.identity(n)
    E[n:, :n] = B
    E[n:, n:] = C

    F = np.zeros((2*n, 2*n-2))
    F[n:, :n-1] = (np.diag(-k * np.ones((n)) / m) + np.diag(k * np.ones((n-1)) / m, -1))[:, :n-1]
    # print("F matrix:\n", F)

    # print("E matrix:\n", E)
    # x0 = np.zeros((n))
    # print("\nx0 vector:", x0)
    # print()
    Y0 = np.concatenate([x0, v0])
    t = np.linspace(0, t_simu, N + 1)
    full_L0 = np.concatenate([L0, np.zeros((n-1))])

    # print("L0 vector:\n", L0)

    def model(Y, t):
        return E @ Y + F @ full_L0

    return t, odeint(model, Y0, t)

def simulate_displacement_wrapper(floe: IceFloe, t_simu=1.0, N=1000):
    """
    Wrapper function for simulate_displacement()
    """
    n = floe.n
    m = floe.m
    k = floe.k
    mu = floe.mu
    v0 = floe.velocities_array()
    x0 = floe.positions_array()
    L0 = floe.initial_lengths()

    return simulate_displacement(n, m, k, mu, x0, v0, L0, t_simu, N)

def simulate_uniform_mov(x0, v, t):
    """
    Simulate the uniform movement of an object of speed v
    """

    def model(y, t):
        return v

    return odeint(model, x0, t)[:, 0]

