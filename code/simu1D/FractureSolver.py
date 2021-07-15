"""
This module defines classes and functions for the collision and displacement of ice floes along with their 2 nodes.
"""
import numpy as np

from PercussionSolver import *

sys.setrecursionlimit(10**4)      # Recursion is important for this problem


""" 
This module only consists of one class: the Fracture class,
which is heavily inspired by the Percussion class. Here we
study the fracture of an ice floe as a Griffith minimization 
problem in 1D. 
"""


class Fracture:
    """
    A class representing a fracture problem between two floes or more
    ice floes. Most of the function implemented in Percussion are
    repeated here.
    """
    def __init__(self, floes, times, n_steps_before_contact, restitution_coef=0.4):

        self.eps = restitution_coef             ## Resitution coefficient for all percussions

        self.t_bef, self.t_aft = times          ## Simulation times before and after first contact
        self.N_bef = n_steps_before_contact     ## Number of time steps before first percussion
        self.N_aft = int(n_steps_before_contact * (self.t_aft / self.t_bef))  ## Number of time steps after first choc

        self.floes = {}                     ## A floe is a mutable list of nodes
        # self.nodes = {}                     ## Nodes must have global ids (same as dict keys)
        # self.springs = {}                   ## Springs must have global ids
        # self.node_neighbors = {}            ## Left and right neighboring nodes for a node (potential collision with
        # them)
        # self.node_neighbors_springs = {}    ## Left and right neighboring springs for a node
        # self.node_parent_floe = {}          ## Id of the ice floe to which this node belongs
        # self.spring_parent_floe = {}        ## Id of the ice floe to which this spring belongs
        # self.spring_neighbors_nodes = {}    ## Left and right neighboring nodes for a spring

        self.confirmation_numbers = {}      ## Point at which all calculations for one floe are confirmed
        self.potential_fractures = {}       ## Potential time step at which fracture occurs
        self.confirmed_fractures = {}       ## Confirmed time step at which fracture occurs
        # self.init_pos = {}                  ## Initial positions for the nodes
        # self.init_vel = {}                  ## Initial velocities for the nodes

        # self.node_radius = {}               ## Radiuses for all nodes
        # self.floe_masses = {}               ## Masses for all nodes
        # self.floe_stiffnesses = {}               ## Stiffnesses for all nodes
        # self.floe_viscosities = {}               ## VIscosities for all nodes
        self.floe_init_lengths = {}              ## Initial lenghts for the springs for all nodes
        self.configurations = {}            ## All observed configurations until the end of simulation

        node_count = 0
        for i, floe in enumerate(floes):
            self.floes[i] = []
            self.floe_masses[i] = floe.m
            self.floe_stiffnesses = floe.k
            self.floe_viscosities = floe.mu
            self.floe_init_lengths = floe.initial_lengths()
            floe_position = floe.positions_array()
            floe_velocities = floe.velocities_array()

            for j, node in enumerate(floe.nodes):
                node.id = node_count                            ## Dangerous ! Not strictly necessary
                # self.nodes[node_count] = node
                self.node_radius[node_count] = node.R
                self.confirmation_numbers[node_count] = 0
                self.node_parent_floe[node_count] = i
                self.init_pos[node_count] = floe_position[i]
                self.init_vel[node_count] = floe_velocities[i]

                if j == 0:
                    self.node_neighbors[node_count] = (None, node_count+1)
                    self.node_neighbors_springs[node_count] = (None, node_count)
                elif j == floe.n - 1:
                    self.node_neighbors[node_count] = (node_count-1, None)
                    self.node_neighbors_springs[node_count] = (node_count-1, None)
                else:
                    self.node_neighbors[node_count] = (node_count - 1, node_count + 1)
                    self.node_neighbors_springs[node_count] = (node_count-1, node_count)


                if self.node_neighbors_springs[node_count][1] == node_count:
                    # self.springs[node_count] = spring
                    self.potential_fractures[node_count] = self.N_bef + self.N_aft
                    self.confirmed_fractures[node_count] = self.N_bef + self.N_aft
                    self.spring_parent_floe[node_count] = i
                    self.spring_neighbors_nodes[node_count] = (node_count, node_count+1)

                self.floes[i].append(node_count)
                node_count += 1

        self.nb_nodes = node_count   ## Total number of nodes

        self.rec_count = {}         ## Recursions depth counter for each collision that happens
        self.contact_indices = {}   ## Each collision has a specific list of contact indices for the percussion


    def could_collide(self, i, j):
        """
        Checks if node of ids i and j could ever collide
        """
        if i is None or j is None:
            return False
        else:
            a, b = min([i, j]), max([i, j])
            are_neighbors = a >= 0 and b < len(self.node_neighbors) and b-a == 1
            spring_not_exists = a not in self.spring_parent_floe
            return are_neighbors and spring_not_exists

    def add_configuration(self):
        """
        Creates and saves a node configuration for our problem (important for plotting)
        """
        self.configurations[len(self.configurations)] = {"ice_floes": self.floes,
                                                         "node_neighbors": self.node_neighbors,
                                                         "node_neighbors_springs": self.node_neighbors_springs,
                                                         "node_parent_floe": self.node_parent_floe,
                                                         "spring_parent_floe": self.spring_parent_floe,
                                                         "spring_neighbors_nodes": self.spring_neighbors_nodes}

    def compute_before_contact(self):
        self.t = np.linspace(0, self.t_bef, self.N_bef+1)
        self.x = np.zeros((self.t.size, self.nb_nodes))      ## Positions for each node
        self.v = self.init_vel * np.ones((self.t.size, self.nb_nodes))      ## Velocities along x for floe1

        for nodes in self.floes.values():
            self.x[:, nodes[0]] = simulate_uniform_mov(self.init_pos[nodes[0]], self.init_vel[nodes[0]], self.t)
            for j in nodes[1:]:
                self.x[:, j] = self.x[:, nodes[0]] + (self.init_pos[j] - self.init_pos[nodes[0]])

    ####### ----------> Fire une boucle ici !
        at_least_one_collision = False
        for node_id, (left, right) in self.node_neighbors.items():
            if self.could_collide(node_id, right) and self.check_colission(node_id, right):
                at_least_one_collision = True
                break

        ## Check whether any two ice floes will collide
        if not at_least_one_collision:
            ## Double simulation time and run phase 1 again
            self.t_bef = self.t_bef * 2
            self.t_aft = self.t_aft * 2
            self.compute_before_contact()
        else:
            ## Then phase 1 is OK
            return


    def check_colission(self, left, right):
        """
        Checks is two floes will collide. If that is the case, save each nodes'
        position and velocity, then discard the remainder of the tensors.
        Returns whether or not there was at least one collision between two nodes.
        """
        start_index = max([self.confirmation_numbers[left], self.confirmation_numbers[right]])
        springs = list(self.node_neighbors_springs[left]) + list(self.node_neighbors_springs[right])
        end_index = min([self.potential_fractures[spring_id] for spring_id in springs])

        collided = False
        col_position = start_index
        for i in range(start_index, end_index):
            if self.x[i, left]+self.node_radius[left].R > self.x[i, right]-self.node_radius[right].R:
                collided = True
                col_position = i
                self.contact_indices[i] = (left, right)
                break

        if collided:
            ## Discard the positions and velocities after collision
            neighbors = self.floes[self.node_parent_floe[left]] + self.floes[self.node_parent_floe[right]]
            self.x[col_position+1:, neighbors ] = np.nan
            self.v[col_position+1:, neighbors ] = np.nan
            self.t[col_position+1:, neighbors ] = np.nan

            ## Update confirmation numbers -- DO IT NOW ?
            for node_id in neighbors:
                self.confirmation_numbers[node_id] = col_position+1

        return collided


    def compute_at_contact(self, i, j):
        """
        Computes the resulting velocities of the two colliding nodes (i=left and j=right)
        """

        if (i is None) or (j is None): return
        # assert self.could_collide(i, j), "These nodes cannot collide, why bring them here ?"

        ## Compute the velocities after contact
        v0 = np.abs(self.v[-1, i])
        v0_ = np.abs(self.v[-1, j])
        m = self.floe_masses[self.node_parent_floe[i]]
        m_ = self.floe_masses[self.node_parent_floe[j]]
        eps = self.eps

        X = eps*(v0 - v0_)
        Y = m*(v0**2) + m_*(v0_**2)
        a = m+m_
        b = 2*m_*X
        c = m_*(X**2) - Y
        Delta = b**2 - 4*a*c
        V01 = (-b - np.sqrt(Delta)) / (2*a)
        V02 = (-b + np.sqrt(Delta)) / (2*a)
        V0 = V01 if V01 >= 0 else V02
        V0_ = V0 + X

        print("VELOCITIES BEFORE/AFTER CONTACT:")
        print(" First floe:", [v0, -np.abs(V0)])
        print(" Second floe:", [-v0_, np.abs(V0_)])

        ## Update velocities at extreme nodes
        self.v[-1, i] = -np.abs(V0)
        self.v[-1, j] = np.abs(V0_)


    def compute_after_contact(self):
        """
        Computes the positions and velocities of the any two colliding floes after a contact
        """
        for node_id, (left, right) in self.node_neighbors.items():
            if self.could_collide(node_id, right) and self.check_colission(node_id, right):

                self.compute_at_contact(node_id, right)       ## Calculate new speeds ...

            ### -----------> Implement these wrappers
                t_sim, xvx1 = self.simulate_displacement_wrapper(self.node_parent_floe[node_id], self.t_aft, self.N_aft)
                t_sim, xvx2 = self.simulate_displacement_wrapper(self.node_parent_floe[right], self.t_aft, self.N_aft)

                c_left, c_right = self.confirmation_numbers[node_id], self.confirmation_numbers[right]
                assert c_left == c_right, "Must have same confirmation numbers"
                self.t[c_left:c_left+self.N_aft] = self.t[-1] + t_sim

                nodes_left = self.floes[self.node_parent_floe[node_id]]
                nodes_right = self.floes[self.node_parent_floe[right]]
                self.x[c_left:c_left+self.N_aft, nodes_left] = xvx1[:, :len(nodes_left)]
                self.x[c_right:c_right+self.N_aft, nodes_right] = xvx2[:, :len(nodes_right)]

                self.v[c_left:c_left+self.N_aft, nodes_left] = xvx1[:, len(nodes_left):]
                self.v[c_right:c_right+self.N_aft, nodes_right] = xvx2[:, len(nodes_right):]

                rec_id = len(self.rec_count)
                print("Recursion depth:", self.rec_count[rec_id])

                ## Check collision then recalculate if applicable
                collided = self.check_colission(node_id, right)
                if (not collided) or (c_left > self.N_bef+self.N_aft) or (self.rec_count[rec_id] > 9800):
                        return
                else:
                    self.rec_count[rec_id] += 1
                    self.compute_after_contact()

            else:
                continue

    def simulate_displacement_wrapper(self, floe_id, t_simu=1.0, N=1000):
        """
        Wrapper function for simulate_displacement(), as a method for ease of use
        """
        n = len(self.floes[floe_id])
        m = self.floe_masses[floe_id]
        k = self.floe_stiffnesses[floe_id]
        mu = self.floe_viscosities[floe_id]
        all_nodes = self.floes[floe_id]
        v0 = self.v[-1, all_nodes]
        x0 = self.v[-1, all_nodes]
        L0 = self.floe_init_lengths[floe_id]

        return simulate_displacement(n, m, k, mu, x0, v0, L0, t_simu, N)
