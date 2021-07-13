"""
This module defines classes and functions for the collision and displacement of ice floes along with their 2 nodes.
"""

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

        self.floes = {}                     ## A floe is a mutable list of nodes
        # self.nodes = {}                     ## Nodes must have global ids (same as dict keys)
        # self.springs = {}                   ## Springs must have global ids
        self.node_neighbors = {}            ## Left and right neighboring nodes for a node (potential collision with
        # them)
        self.node_neighbors_springs = {}    ## Left and right neighboring springs for a node
        self.node_parent_floe = {}          ## Id of the ice floe to which this node belongs
        self.spring_parent_floe = {}        ## Id of the ice floe to which this spring belongs
        self.spring_neighbors_nodes = {}    ## Left and right neighboring nodes for a spring

        self.confirmation_numbers = {}      ## Point at which all calculations for one floe are confirmed
        self.potential_fractures = {}       ## Potential time step at which fracture occurs
        self.confirmed_fractures = {}       ## Confirmed time step at which fracture occurs
        self.init_pos = {}                  ## Initial positions for the nodes

        self.configurations = {}            ## All observed configurations until the end of simulation

        node_count = 0
        for i, floe in enumerate(floes):
            self.floes[i] = []
            floe_position = floe.positions_array()

            for j, node in enumerate(floe.nodes):
                node.id = node_count                            ## Dangerous ! Not strictly necessary
                # self.nodes[node_count] = node
                self.confirmation_numbers[node_count] = 0
                self.node_parent_floe[node_count] = i
                self.init_pos[node_count] = floe_position[i]

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
                    self.potential_fractures[node_count] = None
                    self.confirmed_fractures[node_count] = None
                    self.spring_parent_floe[node_count] = i
                    self.spring_neighbors_nodes[node_count] = (node_count, node_count+1)

                self.floes[i].append(node_count)
                node_count += 1

            # for j, spring in enumerate(floe.springs):
            #
            #     spring.id = spring_count
            #     # self.springs[spring_count] = spring
            #     self.potential_fractures[spring_count] = None
            #     self.spring_parent_floe[spring_count] = i
            #     spring_count += 1

        self.eps = restitution_coef                                             ## Resitution coefficient for all percussions

        self.t_bef, self.t_aft = times                                          ## Simulation times before and after first contact
        self.N_bef = n_steps_before_contact                                  ## Number of time steps before first percussion
        self.N_aft = int(n_steps_before_contact*(self.t_aft/self.t_bef))     ## Number of time steps after first percussion

        self.rec_count = {}         ## Recursions depth counter for each collision that happens
        self.contact_indices = {}   ## Each collision has a specific list of contact indices for the percussion

    def could_collide(self, i, j):
        """
        Checks if node of ids i and j could ever collide
        """
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

