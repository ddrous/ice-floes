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


class Node:
    """
    A class representing one node of an ice floe
    """
    def __init__(self, position, velocity, radius, id_number):
        self.x, self.y = position
        self.vx, self.vy = velocity
        self.R = radius
        self.id = id_number

    def plot(self, figax=None):
        fig, ax = figax

        c = Circle((self.x, self.y), self.R, fc='white', ec='k', lw=2, zorder=10)
        ax.add_patch(c)
        return figax


def d_nodes(node_1, node_2):
    """
    Distance between two nodes
    """
    return np.sqrt((node_1.x - node_2.x) ** 2 + (node_1.y - node_2.y) ** 2)


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
        Ns = 1000

        # Length of the spring
        startx = self.node1.x
        starty = self.node1.y
        R = self.node2.R
        if self.node1.x > self.node2.x:
            startx = self.node2.x
            starty = self.node2.y
            R = self.node1.R
        elif self.node1.x == self.node2.x:
            print("Carefull: zero sized spring!")
        L = d_nodes(self.node2, self.node1) - R
        assert L > 0, "Impossible: Negative length spring"

        # We don't draw coils all the way to the end of the spring: we pad a bit
        ipad1, ipad2 = 150, 150
        w = np.linspace(0, L, Ns)

        # Set up the helix along the x-axis ...
        xp = np.zeros(Ns)
        xp[ipad1:-ipad2] = rs * np.sin(2 * np.pi * ns * w[ipad1:-ipad2] / L)

        # ... then rotate it to align with any desired axis (x-axis).
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta), np.cos(self.theta)]])
        xs, ys = - R @ np.vstack((xp, w))

        ax.plot(startx + xs, starty + (self.D * ys) / np.max(ys), c='k', lw=2)

        return figax


class IceFloe:
    """
    A class representing an ice floe
    """
    def __init__(self, nodes=None, springs=None, id_number=None):
        if nodes:
            self.nodes = nodes
        else:
            # make the nodes randomly
            pass

        self.n_nodes = len(nodes)

        if springs:
            self.springs = springs
        else:
            # make the springs from the nodes available
            self.springs = []
            for i in range(self.n_nodes-1):
                spring = Spring(self.nodes[i],
                               self.nodes[i+1],
                               d_nodes(self.nodes[i], self.nodes[i+1]),
                               (self.nodes[i].R + self.nodes[i+1].R)/2.0,
                               i)
                self.springs.append(spring)

        self.id = id_number

    def update(self, x_array, y_array, vx_array, vy_array):
        """
        Update the position and speeds of all the nodes in the ice floe
        """
        for i, node in enumerate(self.nodes):
            node.x = x_array[i]
            node.y = y_array[i]
            node.vx = vx_array[i]
            node.vx = vy_array[i]

    def plot(self, figax=None):
        """
        Plot an ice floe whose nodes are at (x1,y1) and (x2,y2) with same radius R
        """
        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        #### Put this is percussion problem
        # ax.set_xlim(0, 20)           ## Find min max of all x positions
        # ax.set_ylim(-2 * self.nodes[0].R, 2 * self.nodes[0].R)  ## Find min max of all y
        # ax.set_aspect('equal', adjustable='box')

        for node in self.nodes:
            node.plot(figax)

        for spring in self.springs:
            spring.plot(figax)

        return figax


class Percussion:
    def __init__(self, floe1, floe2):
        pass

    def simulate_displacement(self):
        pass

    def compute_before_contact(self):
        pass

    def compute_at_contact(self):
        pass

    def compute_after_contact(self):
        ## CHech colision then recalculate positions
        pass

    def check_colission(self):
        pass

    def save_fig(self, fps=10, filename="frames/Animation1D.gif", open_file=True):
        """
        Plot both ice floes whose nodes are at (x1,y1) and (x2,y2) with same radius R
        """
        figax = plt.subplots()
        fig, ax = figax
        ## Find
        # ax.set_xlim(0, 20)           ## Find min max of all x positions
        # ax.set_ylim(-2 * self.nodes[0].R, 2 * self.nodes[0].R)  ## Find min max of all y
        # ax.set_aspect('equal', adjustable='box')


        ## For loop to update the floes nodes, then plot


        if open_file:
            ## Open animation
            pass

