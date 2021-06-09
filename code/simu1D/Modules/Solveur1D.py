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

    def plot(self, figax=None):
        fig, ax = figax

        c = Circle((self.x, self.y), self.R, fc='white', ec='k', lw=2, zorder=10)
        ax.add_patch(c)
        return figax




class Spring:
    """
    A class representing one spring of an ice floe
    """
    def __init__(self, node1:Node, node2:Node, initial_length, diameter, id_number):
        self.node1 = node1
        self.node2 = node2
        self.L0 = initial_length
        self.D = diameter
        self.theta = np.arctan2(node2.y-node1.y, node2.x-node1.x) + np.pi/2.0
        self.id = id_number

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
        R = self.node2.R
        if self.node1.x > self.node2.x:
            startx = self.node2.x
            starty = self.node2.y
            R = self.node1.R
        elif self.node1.x == self.node2.x:
            print("Carefull: zero sized spring!")
        L = d_nodes(self.node2, self.node1) - self.node1.R - self.node2.R
        assert L > 0, "Impossible: Negative length spring"

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
    def __init__(self, nodes=None, springs=None, mass=1.0, stiffness=15, viscosity=2.0, rigid_velocity=None, id_number=None):
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
        self.generate_springs()

        print("Generated nodes for ice floe "+str(self.id))

    def generate_springs(self):
        """
        Builds springs and adds them to the floe
        """
        assert self.nodes is not None, "Cannot build springs if nodes are absent."

        self.springs = []
        for i in range(self.n - 1):
            spring = Spring(self.nodes[i],
                            self.nodes[i + 1],
                            d_nodes(self.nodes[i], self.nodes[i + 1]),
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
    def __init__(self, floe1:IceFloe, floe2:IceFloe, time_before_contact=4.0, time_at_contact=1.0, time_after_contact=16.0, n_steps_before_contact=2000, restitution_coef=0.4):
        self.floe1, self.floe2 = floe1, floe2
        self.eps = restitution_coef

        self.t_bef, self.t_at, self.t_aft = time_before_contact, time_at_contact, time_after_contact
        self.N_bef = n_steps_before_contact
        self.N_at = n_steps_before_contact//10
        self.N_aft = int(n_steps_before_contact*(time_after_contact/time_before_contact))

        self.rec_count = 0      ## Recursion depth counter

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
            self.t_bef = self.t_bef*2
            self.compute_before_contact()
        else:
            ## Then phase 1 is OK
            return

    def compute_at_contact(self):
        """
        Computes the resulting velocities of the two colliding nodes
        """
        t_con, xvx1 = simulate_displacement_wrapper(self.floe1, self.t_at, self.N_at)
        t_con, xvx2 = simulate_displacement_wrapper(self.floe2, self.t_at, self.N_at)

        print("n compare", self.floe2.n, len(xvx2[0,:]))
        ## Compute the integrand for speed calculation
        intgr = self.floe1.k*(xvx1[:,self.floe1.n-2] - xvx1[:,self.floe1.n-1]) + self.floe1.mu*(xvx1[:,-2] - xvx1[:,-1]) \
                - self.floe2.k*(xvx2[:,0] - xvx2[:,1]) - self.floe2.mu*(xvx2[:,self.floe2.n] - xvx2[:,self.floe2.n+1])
        I = np.trapz(y=intgr, x=t_con)
        print("Value of I for computation:", I)

        ## Compute the velocities after contact
        v0 = np.abs(self.floe1.nodes[-1].vx)
        v0_ = np.abs(self.floe2.nodes[0].vx)
        m = self.floe1.m
        m_ = self.floe2.m
        eps = self.eps

        V0 = (I + (m - eps * m) * v0 + (1 + eps) * m * v0_) / (m + m_)
        V0_ = (I + (1 + eps) * m * v0 + (m_ - eps * m) * v0_) / (m + m_)
        print("VELOCITIES BEFORE/AFTER CONTACT:")
        print(" First floe:", [v0, -np.abs(V0)])
        print(" Second floe:", [-np.abs(V0_), v0_])

        ## Update velocities at extreme nodes
        self.floe1.nodes[-1].vx = -np.abs(V0)
        self.floe2.nodes[0].vx = np.abs(V0_)

    def compute_after_contact(self):
        t_sim, xvx1 = simulate_displacement_wrapper(self.floe1, self.t_aft, self.N_aft)
        t_sim, xvx2 = simulate_displacement_wrapper(self.floe2, self.t_aft, self.N_aft)

        self.t = np.concatenate([self.t, self.t[-1] + t_sim])

        self.x1 = np.concatenate([self.x1, self.x1[-1,:] + xvx1[:, :self.floe1.n]])
        self.v1 = np.concatenate([self.v1, xvx1[:, self.floe1.n:]])

        self.x2 = np.concatenate([self.x2, self.x2[-1,:] + xvx2[:, :self.floe2.n]])
        self.v2 = np.concatenate([self.v2, xvx2[:, self.floe2.n:]])

        print("Recursion depth:", self.rec_count)

        ## Check collision then recalculate if applicable
        collided = self.check_colission(self.contact_index+2)
        if (not collided) or (self.t.size > self.N_bef+self.N_aft) or (self.rec_count > 980):
        # if (not collided) or (self.t.size > 2009):
                return
        else:
            self.rec_count += 1
            self.compute_at_contact()
            self.compute_after_contact()


    def check_colission(self, start_index=0):
        """
        Checks is the two floes will collide. If that is the case, save each nodes'
        position and velocity, then discard the remainder of the tensors.
        """
        assert start_index < self.t.size, "Starting index to check collision too big"

        self.contact_index = -1
        for i in range(start_index, self.t.size):
            if self.x1[i,-1]+self.floe1.nodes[-1].R > self.x2[i,0]-self.floe2.nodes[0].R:

                ## If collision, save each nodes positions and speed
                for j, node in enumerate(self.floe1.nodes):
                    node.x = self.x1[i,j]
                    node.vx = self.v1[i,j]

                for j, node in enumerate(self.floe2.nodes):
                    node.x = self.x2[i,j]
                    node.vx = self.v2[i,j]

                self.contact_index = i
                break

        if self.contact_index != -1:
            ## Discard the positions and velocities after collision
            self.x1 = self.x1[:self.contact_index+1]
            self.v1 = self.v1[:self.contact_index+1]
            self.x2 = self.x2[:self.contact_index+1]
            self.v2 = self.v2[:self.contact_index+1]
            self.t = self.t[:self.contact_index+1]

        return self.contact_index != -1

    def plot_momentum(self):
        pass

    def plot_energy(self):
        pass

    def save_fig(self, fps=24, filename="Animation1D.gif", open_file=True):
        """
        Plot both ice floes whose nodes are at (x1,y1) and (x2,y2) with same radius R
        """
        min_X = self.floe1.nodes[0].x0 - self.floe1.nodes[0].R
        max_X = self.floe2.nodes[-1].x0 + self.floe2.nodes[-1].R
        max_R = np.max([self.floe1.max_radius(), self.floe2.max_radius()])

        fig = plt.figure(figsize=(max_X-min_X, 5*max_R), dpi=72)
        ax = fig.add_subplot(111)

        # ax.set_xlim(min_X, max_X)
        # ax.set_ylim(-4 * max_R, 4 * max_R)
        # ax.set_aspect('equal', adjustable='box')

        dt = self.t_bef/self.N_bef
        di = int(1 / fps / dt)

        img_list = []

        print("Generating "+str(di)+" frames ...")
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
        print("OK! saved file at "+filename)

        if open_file:
            ## Open animation
            os.system('gthumb '+filename)     ## Only on Linux










""" General purpose functions. The function simulate_displacement is the most heart of the problem """

def d_nodes(node_1, node_2):
    """
    Distance between two nodes
    """
    return np.sqrt((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2)

def fig2img(fig):
    """ Convert a Matplotlib figure to a PIL Image and return it """
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = PILImage.open(buf)
    return img

def simulate_displacement(n=2, m=1.0, k=18.0, mu=1.3, v0=None, t_simu=1.0, N=1000):
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

    # print("E matrix:\n", E)

    Y0 = np.concatenate([np.zeros((n)), v0])
    t = np.linspace(0, t_simu, N + 1)

    def model(Y, t):
        return E @ Y

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
    return simulate_displacement(n, m, k, mu, v0, t_simu, N)

def simulate_uniform_mov(x0, v, t):
    """
    Simulate the uniform movement of an object of speed v
    """

    def model(y, t):
        return v

    return odeint(model, x0, t)[:, 0]

