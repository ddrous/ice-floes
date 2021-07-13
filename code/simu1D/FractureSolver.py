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
        self.N_aft = int(n_steps_before_contact * (self.t_aft / self.t_bef))  ## Number of time steps after first percussion

        self.floes = {}                     ## A floe is a mutable list of nodes
        # self.nodes = {}                     ## Nodes must have global ids (same as dict keys)
        # self.springs = {}                   ## Springs must have global ids
        self.node_neighbors = {}            ## Left and right neighboring nodes for a node (potential collision with
        # them)
        self.node_radius = {}               ## Radiuses for all nodes
        self.node_neighbors_springs = {}    ## Left and right neighboring springs for a node
        self.node_parent_floe = {}          ## Id of the ice floe to which this node belongs
        self.spring_parent_floe = {}        ## Id of the ice floe to which this spring belongs
        self.spring_neighbors_nodes = {}    ## Left and right neighboring nodes for a spring

        self.confirmation_numbers = {}      ## Point at which all calculations for one floe are confirmed
        self.potential_fractures = {}       ## Potential time step at which fracture occurs
        self.confirmed_fractures = {}       ## Confirmed time step at which fracture occurs
        self.init_pos = {}                  ## Initial positions for the nodes
        self.init_vel = {}                  ## Initial velocities for the nodes

        self.configurations = {}            ## All observed configurations until the end of simulation

        node_count = 0
        for i, floe in enumerate(floes):
            self.floes[i] = []
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

        ## Check whether any two ice floes will collide
        collided = self.check_colission()
        if not collided:
            ## Double simulation time and run phase 1 again
            self.t_bef = self.t_bef * 2
            self.t_aft = self.t_aft * 2
            self.compute_before_contact()
        else:
            ## Then phase 1 is OK
            return


    def check_colission(self):
        """
        Checks is any two floes will collide. If that is the case, save each nodes'
        position and velocity, then discard the remainder of the tensors.
        Returns whether or not there was at least one collision between two nodes.
        """
        at_least_one_collision = False
        for node_id, (left, right) in self.node_neighbors.items():
            if self.could_collide(node_id, right):
                start_index = max([self.confirmation_numbers[node_id], self.confirmation_numbers[right]])
                springs = list(self.node_neighbors_springs[node_id]) + list(self.node_neighbors_springs[right])
                end_index = min([self.potential_fractures[spring_id] for spring_id in springs])
            else:
                continue

            collided = False
            col_position = start_index
            for i in range(start_index, end_index):
                if self.x[i, node_id]+self.node_radius[node_id].R > self.x[i, right]-self.node_radius[right].R:
                    collided = True
                    col_position = i
                    self.contact_indices[i] = (node_id, right)
                    break

            if collided:
                ## Discard the positions and velocities after collision
                self.x[col_position+ 1: ] = np.nan
                self.v[col_position+ 1: ] = np.nan
                self.t[col_position+ 1: ] = np.nan

            at_least_one_collision = at_least_one_collision or collided
        return at_least_one_collision
