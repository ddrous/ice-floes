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
    def __init__(self, floes, times, nStepsBefContact, restitutionCoef=0.4):

        self.eps = restitutionCoef              ## Resitution coefficient for all percussions

        self.tBef, self.tAft = times            ## Simulation times before and after first contact
        self.NBef = nStepsBefContact            ## Number of time steps before first percussion
        self.NAft = int(nStepsBefContact * (self.tAft / self.tBef))  ## Number of time steps after first choc

        self.floes = {}
        for i, floe in enumerate(floes):
            floe.id = i
            self.floes[i] = floe
        self.nodeIds = []                     ## A list of all node ids from all different floes
        self.springIds = []                   ## A list of all spring ids from all different floes

        self.confirmationNumbers = {}      ## Point at which all calculations for one floe are confirmed
        self.potentialFractures = {}       ## Potential time step at which fracture occurs
        self.confirmedFractures = {}       ## Confirmed time step at which fracture occurs

        nodeCount = 0
        for i, floe in self.floes.items():
            self.floes[i] = floe

            for j, node in enumerate(floe.nodes):
                node.id = nodeCount
                self.confirmationNumbers[nodeCount] = 0
                node.parentFloe = floe.id
                self.nodeIds.append(nodeCount)

                # if j == 0:
                #     node.leftNode, node.rightNode = (None, nodeCount+1)
                #     node.leftSpring, node.rightSpring = (None, nodeCount)
                # elif j == floe.n - 1:
                #     node.leftNode, node.rightNode = (nodeCount-1, None)
                #     node.leftSpring, node.rightSpring = (nodeCount-1, None)
                # else:
                #     node.leftNode, node.rightNode = (nodeCount - 1, nodeCount + 1)
                #     node.leftSpring, node.rightSpring = (nodeCount-1, nodeCount)

                node.leftNode, node.rightNode = (nodeCount - 1, nodeCount + 1)
                node.leftSpring, node.rightSpring = (nodeCount - 1, nodeCount)

                if j == 0:
                    if i == 0:
                        node.leftNode, node.rightNode = (None, nodeCount + 1)
                    node.leftSpring, node.rightSpring = (None, nodeCount)
                elif j == floe.n - 1:
                    if i == len(self.floes) - 1:
                        node.leftNode, node.rightNode = (nodeCount - 1, None)
                    node.leftSpring, node.rightSpring = (nodeCount-1, None)


                if node.rightSpring == nodeCount:       ## Just a random test to get good ids!
                    floe.springs[j].id = nodeCount
                    floe.springs[j].parentFloe = i
                    floe.springs[j].leftNode = nodeCount
                    floe.springs[j].rigthNode = nodeCount + 1
                    self.potentialFractures[nodeCount] = self.NBef + self.NAft - 2
                    self.confirmedFractures[nodeCount] = self.NBef + self.NAft - 2
                    self.springIds.append(nodeCount)

                nodeCount += 1

        self.nbNodes = nodeCount   ## Total number of nodes

        self.initPos = self.positionsArray()                  ## Initial positions for the nodes
        self.initVel = self.velocitiesArray()                 ## Initial velocities for the nodes
        self.initLengths = self.initialLengths()              ## Initial lenghts for the springs for all nodes

        self.recCount = {}         ## Recursions depth counter for each collision that happens
        self.contactIndices = {}   ## Each collision has a specific list of contact indices for the percussion

        self.configurations = {}            ## All observed configurations until the end of simulation
        self.configurations[0] = self.floes.copy()
        print("PRINT THE INITIAL CONFIG CONFIG ", self.configurations)

    def printDetails(self):
        """
        Prints details about each node in the problem
        """
        print("\nFRACTURE PROBLEM PARAMETERS")
        print("  times:", (self.tBef, self.tAft))
        print("  nb steps:", (self.NBef, self.NAft))
        print("  restitution coefficient:", self.eps)
        print()
        for floe in self.floes.values():
            for node in floe.nodes:
                print(node)

    def positionsArray(self):
        """
        Returns positions of all nodes of all floes
        """
        x = np.zeros((self.nbNodes))
        for floe in self.floes.values():
            for node in floe.nodes:
                x[node.id] = node.x
        return x


    def velocitiesArray(self):
        """
        Returns velocities of all nodes in a specific order
        """
        v = np.zeros((self.nbNodes))
        for floe in self.floes.values():
            for j, node in enumerate(floe.nodes):
                v[node.id] = node.vx
        return v


    def initialLengths(self):
        """
        Initial lengths of springs of all floes
        """
        L0 = []
        for floe in self.floes.values():
            for spring in floe.springs:
                L0.append(spring.L0)
        return np.array(L0)

    def locateNodeId(self, nodeId):
        """
        Tells you to witch floe a node belongs to and its local id
        """
        for i, floe in self.floes.items():
            for j, node in enumerate(floe.nodes):
                if node.id == nodeId:
                    return (i, j)
        raise IndexError("A node with the id ["+str(nodeId)+"] does not exist")

    def locateNode(self, nodeId):
        """
        Returns the node corresponding to a certain id
        """
        c = self.locateNodeId(nodeId)
        return self.floes[c[0]].nodes[c[1]]

    def neighbouringNodes(self, nodeId):
        """
        Tells you neighboring nodes for a particular node id
        """
        node = self.locateNode(nodeId)     ## Local coordinates of the node
        return (node.leftNode, node.rightNode)

    def neighbouringSprings(self, nodeId):
        """
        Tells you neighboring springs for a particular node id
        """
        node = self.locateNode(nodeId)     ## Local coordinates of the node
        return (node.leftSpring, node.rightSpring)


    def computeBeforeContact(self):
        self.t = np.linspace(0, self.tBef, self.NBef+self.NAft+1)
        initPos = self.positionsArray()
        initVel = self.velocitiesArray()
        self.x = initPos * np.ones((self.t.size, self.nbNodes))                              ## Positions for each node
        self.v = initVel * np.ones((self.t.size, self.nbNodes))      ## Velocities along x

        # print("Time HERE:", self.t)
        for floe in self.floes.values():
            firstNode = floe.nodes[0].id
            self.x[:, firstNode] = simulate_uniform_mov(initPos[firstNode], initVel[firstNode], self.t)
            for node in floe.nodes[1:]:
                self.x[:, node.id] = self.x[:, firstNode] + (self.initPos[node.id] - self.initPos[firstNode])

        atLeastOneCollision = False
        for floe in self.floes.values():
            for node in floe.nodes:
                if self.checkCollision(node.id, node.rightNode):
                    atLeastOneCollision = True
                    break
            if atLeastOneCollision:
                break

        ## Check whether any two ice floes will collide
        if not atLeastOneCollision:
            ## Double simulation time and run phase 1 again
            self.tBef = self.tBef * 2
            self.tAft = self.tAft * 2
            self.computeBeforeContact()
        else:
            ## Then phase 1 is OK
            return

    def couldCollide(self, left, right):
        """
        Checks if node of ids i and j could ever collide
        """
        if left is None or right is None:
            return False
        else:
            a, b = min([left, right]), max([left, right])
            areNeighbors = a >= 0 and b < self.nbNodes and b-a == 1
            # springNotExists = a not in self.springIds
            nL, nR = self.locateNode(left), self.locateNode(right)
            differentParents = nL.parentFloe != nR.parentFloe
            return areNeighbors and differentParents


    def checkCollision(self, left, right):
        """
        Checks is two floes will collide. If that is the case, save each nodes'
        position and velocity, then discard the remainder of the tensors.
        Returns whether or not there was at least one collision between two nodes.
        """

        if not self.couldCollide(left, right):
            return False
        else:
            startIndex = max([self.confirmationNumbers[left], self.confirmationNumbers[right]])
            springs = list(self.neighbouringSprings(left)) + list(self.neighbouringSprings(right))
            endIndex = min([self.potentialFractures[springId] for springId in springs if springId])

            leftNode, rightNode = self.locateNode(left), self.locateNode(right)
            collided = False
            colPosition = startIndex
            for i in range(startIndex, endIndex):
                if self.x[i, left]+leftNode.R > self.x[i, right]-rightNode.R:
                    collided = True
                    colPosition = i
                    self.contactIndices[i] = (left, right)
                    break

            if collided:
                ## Discard the positions and velocities after collision
                neighbors = list(self.neighbouringNodes(left)) + list(self.neighbouringNodes(right))
                self.x[colPosition+1:, neighbors ] = np.nan
                self.v[colPosition+1:, neighbors ] = np.nan
                self.t[colPosition+1:] = np.nan

                ## Update confirmation numbers -- DO IT NOW ?
                for nodeId in neighbors:
                    self.confirmationNumbers[nodeId] = colPosition

            return collided


    def computeAtContact(self, left, right):
        """
        Computes the resulting velocities of the two colliding nodes (i=left and j=right)
        """
        if (left is None) or (right is None): return
        leftNode, rightNode = self.locateNode(left), self.locateNode(right)
        # assert self.could_collide(i, j), "These nodes cannot collide, why bring them here ?"

        ## Compute the velocities after contact
        v0 = np.abs(leftNode.vx)
        v0_ = np.abs(rightNode.vx)
        m = self.floes[leftNode.parentFloe].m
        m_ = self.floes[rightNode.parentFloe].m
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
        leftNode.vx = -np.abs(V0)
        rightNode.vx = np.abs(V0_)


    def computeAfterContact(self):
        """
        Computes the positions and velocities of the any two colliding floes after a contact
        """

        for floe in self.floes.values():
            for node in floe.nodes:
                left, right = node.id, node.rightNode
                try:
                    leftNode, rightNode = self.locateNode(left), self.locateNode(right)
                except IndexError:
                    print("The got and error. SORRY Y'LL !", left, right)
                    continue

                # if self.couldCollide(left, right) and self.checkCollision(left, right):
                if self.checkCollision(left, right):
                    print("I HAVE GOTTEN IN HERE MY MAN")

                    self.computeAtContact(left, right)       ## Calculate new speeds ...

                    tSim, xvxL = simulate_displacement_wrapper(self.floes[leftNode.parentFloe], self.tAft, self.NAft)
                    tSim, xvxR = simulate_displacement_wrapper(self.floes[rightNode.parentFloe], self.tAft, self.NAft)

                    cL, cR = self.confirmationNumbers[left], self.confirmationNumbers[right]
                    assert cL == cR, "Must have same confirmation numbers"

                    print("TSIM size:", self.t.size, "After:", self.NAft, "Confirm left:", cL)
                    end1, end2 = cL + self.NAft, cL + self.NAft - cL
                    if cL + self.NAft > self.t.size:
                        end1, end2 = self.t.size, self.t.size - cL

                    self.t[cL:end1] = (self.t[cL] + tSim)[:end2]

                    floeL = self.floes[leftNode.parentFloe]
                    nodeIdsL = [node.id for node in floeL.nodes]
                    floeR = self.floes[rightNode.parentFloe]
                    nodeIdsR = [node.id for node in floeR.nodes]
                    self.x[cL:end1, nodeIdsL] = xvxL[:end2, :floeL.n]
                    self.x[cR:end1, nodeIdsR] = xvxR[:end2, :floeR.n]

                    self.v[cL:end1, nodeIdsL] = xvxL[:end2, floeL.n:]
                    self.v[cR:end1, nodeIdsR] = xvxR[:end2, floeR.n:]

                    recId = len(self.recCount)
                    print("Recursion depth:", self.recCount[recId])

                    ## Check collision then recalculate if applicable
                    collided = self.checkCollision(left, right)
                    if (not collided) or (cL > self.NBef+self.NAft) or (self.recCount[recId] > 980):
                            return
                    else:
                        self.recCount[recId] += 1
                        self.computeAfterContact()

                else:
                    continue



    def saveFig(self, fps=24, filename="Exports/Anim1D.gif", openFile=True):
        """
        Plot both ice floes whose nodes are at (x1,y1) and (x2,y2) with same radius R
        """
        leftMostNode = self.locateNode(0)
        rightMostNode = self.locateNode(self.nbNodes-1)
        min_X = leftMostNode.x0 - leftMostNode.R
        max_X = rightMostNode.x0 + rightMostNode.R
        max_R = np.max([floe.max_radius() for floe in self.floes.values()])

        plt.style.use("default")
        fig = plt.figure(figsize=(max_X-min_X, 5*max_R), dpi=72)
        ax = fig.add_subplot(111)

        # ax.set_xlim(min_X, max_X)
        # ax.set_ylim(-4 * max_R, 4 * max_R)
        # ax.set_aspect('equal', adjustable='box')

        dt = self.tBef/self.NBef
        di = int(1 / fps / dt)

        img_list = []

        print("Generating frames ...")
        ## For loop to update the floes nodes, then plot
        for i in range(0, self.t.size, di):

            ##----------------------------------------
            for index in self.configurations.keys():
                if i >= index:
                    floes = self.configurations[index]
                    break
            ##----------------------------------------

            print("  ", i // di, '/', self.t.size // di)
            for floe in self.floes.values():
                for node in floe.nodes:
                    node.x = self.x[i, node.id]
                    node.vx = self.v[i, node.id]
                floe.plot(figax=(fig,ax))

            ax.set_xlim(min_X, max_X)
            ax.set_ylim(-2 * max_R, 2 * max_R)
            ax.set_aspect('equal', adjustable='box')

            img_list.append(fig2img(fig))           ### Use tight borders !!!!!

            plt.cla()      # Clear the Axes ready for the next image.

        imageio.mimwrite(filename, img_list)
        print("OK! saved file '"+filename+"'")

        if openFile:
            ## Open animation
            os.system('gthumb '+filename)     ## Only on Linux
