"""
This module defines classes and functions for the collision and displacement of ice floes along with their 2 nodes.
"""
import numpy as np

from PercussionSolver import *
from threading import Barrier, Thread
from copy import deepcopy
from sys import stdout          ## <<-- Only for printing immediately!!!

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

        self.simuTimeStep = self.NAft  ## ==NAft        ## Tne number of time steps we simulate at a time
        self.tSim = self.tAft * self.simuTimeStep / self.NAft

        self.floes = {}
        for i, floe in enumerate(floes):
            floe.id = i
            self.floes[i] = floe

        nodeCount = 0
        for i, floe in self.floes.items():
            self.floes[i] = floe

            for j, node in enumerate(floe.nodes):
                node.id = nodeCount
                node.parentFloe = floe.id

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

                nodeCount += 1

        self.nbNodes = nodeCount   ## Total number of nodes

        self.initPos = self.positionsArray()                  ## Initial positions for the nodes
        self.initVel = self.velocitiesArray()                 ## Initial velocities for the nodes
        self.initLengths = self.initialLengths()              ## Initial lenghts for the springs for all nodes

        self.collCount = {}         ## Recursions depth counter for each collision that happens

        self.configurations = {}            ## All observed configurations until the end of simulation
        self.configurations[0] = deepcopy(self.floes)

        self.checkCollFrom = {}             ## Position from which to perform collision and fracture checks
        self.checkFracFrom = {}             ## Position from which to perform collision and fracture checks

        self.collLoc = []

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
                print(node.get_details())

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

    def locateSpringId(self, springId):
        """
        Tells you to witch floe a spring belongs to and its local id
        """
        for i, floe in self.floes.items():
            for j, spring in enumerate(floe.springs):
                if spring.id == springId:
                    return (i, j)
        raise IndexError("A spring with the id ["+str(springId)+"] does not exist")

    def locateSpring(self, springId):
        """
        Returns the spring corresponding to a certain id
        """
        c = self.locateNodeId(springId)
        return self.floes[c[0]].springs[c[1]]

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

    def neighbouringSpringEdges(self, springId):
        """
        Tells you neighboring nodes (edges) for a particular spring id
        """
        spring = self.locateSpring(springId)     ## Local coordinates of the node
        return (spring.leftNode, spring.rightNode)


    def computeBeforeContact(self):
        self.t = np.linspace(0, self.tBef, self.NBef+1)
        initPos = self.positionsArray()
        initVel = self.velocitiesArray()
        self.x = initPos * np.ones((self.t.size, self.nbNodes))      ## Positions for each node
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
            self.tBef = self.tBef*2
            self.tAft = self.tAft*2
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
            self.checkCollFrom[(left, right)] = self.t.size
            return False

        else:
            startIndex = self.checkCollFrom.setdefault((left, right), 0)
            # startIndex = min(self.checkFrom.values()) if len(self.checkFrom) > 0 else 0
            endIndex = self.t.size

            # if startIndex >= endIndex:
            #     # startIndex = endIndex-1
            #     self.checkCollFrom[(left, right)] = endIndex-1
            #     return False

            leftNode, rightNode = self.locateNode(left), self.locateNode(right)

            for i in range(startIndex, endIndex):

                # diffVel = np.abs(self.v[i, left]-self.v[i, right]) > 1e-2
                # collCount = self.collCount.setdefault((left, right), 0)
                # if self.x[i, left]+leftNode.R > self.x[i, right]-rightNode.R and collCount < 100:

                if self.x[i, left] + leftNode.R > self.x[i, right] - rightNode.R:

                    ## If collision, save each nodes positions and speed
                    for floe in self.floes.values():
                        for node in floe.nodes:
                            node.x = self.x[i-1, node.id ]
                            node.vx = self.v[i-1, node.id ]

                    ## Discard the positions and velocities after collision
                    self.x = self.x[:i-1, :]
                    self.v = self.v[:i-1, :]
                    self.t = self.t[:i-1]

                    self.checkCollFrom[(left, right)] = i

                    self.computeAtContact(left, right)

                    return True

            ## We only get here if there was no collision
            self.checkCollFrom[(left, right)] = self.t.size
            return False


    def computeAtContact(self, left, right):
        """
        Computes the resulting velocities of the two colliding nodes (i=left and j=right)
        """
        if (left is None) or (right is None): return
        leftNode, rightNode = self.locateNode(left), self.locateNode(right)
        # assert self.could_collide(i, j), "These nodes cannot collide, why bring them here ?"
        self.collCount[(left, right)] = self.collCount.setdefault((left, right), 0) + 1

        ## Compute the velocities after contact
        v0 = leftNode.vx
        v0_ = rightNode.vx
        m = self.floes[leftNode.parentFloe].m
        m_ = self.floes[rightNode.parentFloe].m
        eps = self.eps

        ###########     TROISIÈME ALTERNATIVE     ####################
        # v0 = np.abs(leftNode.vx)
        # v0_ = np.abs(rightNode.vx)
        # X = eps*(v0 - v0_)
        # Y = m*(v0**2) + m_*(v0_**2)
        # a = m+m_
        # b = 2*m_*X
        # c = m_*(X**2) - Y
        # Delta = b**2 - 4*a*c
        # V01 = (-b - np.sqrt(Delta)) / (2*a)
        # V02 = (-b + np.sqrt(Delta)) / (2*a)
        #
        # V0 = V01 if V01 >= 0 else V02
        # V0_ = V0 + X
        #############################################################

        ###########     QUATRIÈME ALTERNATIVE       ####################
        ### Conservation classice de l'Ec et de la qte de mouv
        # V0 = (2*m_*v0_ + (m-m_) * v0) / (m + m_)
        # V0_ = (2 * m * v0 + (m_ - m) * v0_) / (m + m_)
        ################################################################

        ###########      CINQUIEME ALTERNATIVE       ####################
        ## Ici on utilise la page 37 du brouillon avec l'impulsion
        ### V0 = (-m*eps*np.abs(v0-v0_) + m*v0 + m_*v0_) / (m+m_)
        ### V0_ = (m_*eps*np.abs(v0-v0_) + m*v0 + m_*v0_) / (m+m_)
        # V0 = v0 - (m_ * (1.0+eps) * (v0-v0_)) / (m + m_)
        # V0_ = v0_ + (m * (1.0+eps) * (v0-v0_)) / (m + m_)
        # if self.collCount[(left, right)] > 100:
        #     eps = 1.0
        #     self.collCount[(left, right)] = 100
        #     V0 = -np.abs(v0 - (m_ * (1.0+eps) * (v0-v0_)) / (m + m_))
        #     V0_ = np.abs(v0_ + (m * (1.0+eps) * (v0-v0_)) / (m + m_))
        ##################################################################

        ###########     SIXIÈME ALTERNATIVE       ####################
        ## Ici on améliore la technique des jeux vidéos
        leftDist = leftNode.x - self.locateNode(leftNode.leftNode).x
        leftForce = self.floes[leftNode.parentFloe].k * (self.locateSpring(leftNode.leftSpring).L0 - leftDist)

        rightDist = self.locateNode(rightNode.rightNode).x - rightNode.x
        rightForce = self.floes[rightNode.parentFloe].k * (rightDist - self.locateSpring(rightNode.rightSpring).L0)

        print("Dist", leftDist, rightDist)
        print("Force", leftForce, rightForce)

        deltaTStar = 0.01            ## Temps que dure la collision
        L = deltaTStar * leftForce
        R = deltaTStar * rightForce

        I = ((v0 - v0_)*(1.0 + eps) + (L/m)-(R/m_)) / ((1.0/m) + (1.0/m_))

        V0 = v0 + (L-I)/m
        V0_ = v0_ + (R+I)/m_
        if self.collCount[(left, right)] > 100:
            eps = 1.0
            self.collCount[(left, right)] = 100
            V0 = -np.abs(v0 - (m_ * (1.0+eps) * (v0-v0_)) / (m + m_))
            V0_ = np.abs(v0_ + (m * (1.0+eps) * (v0-v0_)) / (m + m_))
        ##################################################################

        print("\nCONTACT ("+str(left)+", "+str(right)+") OCCURRED, VELOCITIES ARE:")
        print("   Left node: ", [v0, V0])
        print("   Right node:", [v0_, V0_])
        print("Collision count for (" + str(left) + ", " + str(right) + "):", self.collCount[(left, right)])
        print("Current time size:", self.t.size)

        ## Update velocities at extreme nodes
        leftNode.vx = V0
        rightNode.vx = V0_
        # leftNode.vx = -np.abs(V0)
        # rightNode.vx = np.abs(V0_)


    def computeAfterContact(self):
        """
        Computes the positions and velocities of the any two colliding floes after a contact
        """
        # for floe in self.floes.values():
            # for node in floe.nodes:
                # left, right = node.id, node.rightNode

                # if self.checkCollision(left, right):

                # self.computeAtContact(left, right)       ## Calculate new speeds ...

        x = np.zeros((self.simuTimeStep+1, self.nbNodes))
        vx = np.zeros((self.simuTimeStep+1, self.nbNodes))

        for floe in self.floes.values():
            tNew, xvx = simulate_displacement_wrapper(floe, self.tSim, self.simuTimeStep)

            globalIds = [node.id for node in floe.nodes]
            x[:, globalIds], vx[:, globalIds] = xvx[:, :floe.n], xvx[:, floe.n:]

        try:
            self.x = np.concatenate([self.x, x])
        except ValueError:                                  ####<<--clean this !
            print("hey got ya")
        self.v = np.concatenate([self.v, vx])
        self.t = np.concatenate([self.t, self.t[-1] + tNew])

        return


    def fractureEnergy(self, floeId, brokenSprings=None):
        """
        Computes the fracture energy of a fractured ice floe, i.e. some springs and broken.
        """
        ### Bien préciser qu'on est en déformation élastqiue: et donc la longueur de la fracture est la longeur initiale des ressorts

        try:
            floe = self.floes[floeId]
        except IndexError:
            print("Error: Ice floe of id "+str(floeId)+"does not exist (yet).")
            return 0

        brokenLength = 0.0
        for i in brokenSprings:
            try:
                brokenLength += floe.springs[i].L0
            except IndexError:
                print("Error: Spring id "+str(i)+" is not a valid id for ice floe "+str(floeId))

        return floe.L * brokenLength



    def deformationEnergy(self, floeId=None, brokenSprings=None, end=0):
        """
        Computes the deformation energy (sum of the elastic energy and the dissipated
        energy) when the ice floe is fractured, i.e. some springs and broken.
        """

        try:
            floe = self.floes[floeId]
        except IndexError:
            print("Error: Ice floe of id "+str(floeId)+" does not exist (yet).")
            return 0

        ## Not very sure <<<--- FIX THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if end >= self.t.size:
            return 0
        ##--------------------------------------

        ## Energie potentielle elastique (au pas de temps `end`)
        k = np.full((floe.n - 1), floe.k)
        k[[i for i in brokenSprings if i]] = 0.0         ## Eliminate the broken springs

        firstId, lastId = floe.nodes[0].id, floe.nodes[-1].id
        # print("\nK, FIRST AND LAST ID", len(k), firstId, lastId)
        # print()
        # stdout.flush()
        # try:
        E_el = 0.5 * np.sum(k * (self.x[end, firstId+1:lastId+1] - self.x[end, firstId:lastId]
                               - floe.init_lengths)**2, axis=-1)
        # except ValueError:
        #     print("Ahh Haaa: GOT YOU")

        ## Energie dissipée entre les temps `0` et `end`
        # mu = np.full((floe.n - 1), floe.mu)
        # mu[brokenSprings] = 0.0        ## Eliminate the broken dampers
        #
        # integrand = np.sum(mu * (self.v[:end+1, firstId+1:lastId+1] - self.v[:end+1, firstId:lastId])**2, axis=-1)
        # t = self.t[1:end+1] - self.t[:end]
        # E_r = np.sum(integrand*t)
        #
        # return E_el + E_r     #### ---- STUDY THIS PART AGAIN ---- ####
        # print("Def energy:", E_el)
        return E_el


    def griffithMinimization(self, floeId=None):
        """
        Studies the fracture problem to see if it is worth addind a path to the crack.
        The total energy here is the sum of the deformation energy and the fracture energy.
        """

        ## Specify the time steps for the computations
        floe = self.floes[floeId]

        if floe.n <= 3:
            return False, None, ()        ## Only big floes can be fractured

        oldEnd = self.checkFracFrom[floeId]        ## Time step at witch we compute the current energy
        # startCheckPos = []
        # for node in floe.nodes:
        #     startCheckPos.append(self.checkFracFrom.setdefault((node.id, node.rightNode), 0))
        # oldEnd = min(startCheckPos)         ## Time step at witch we compute the current energy

        stopCheckAt = self.t.size         ## Time step at witch we compute the current energy

        ## Compute the current energy of the system (no broken spring)
        oldBrokenSprings = []
        try:
            defEn = self.deformationEnergy(floeId, oldBrokenSprings, oldEnd)
        except IndexError:
            print("Got you!")       ### <-- check here
        fracEn = self.fractureEnergy(floeId, oldBrokenSprings)
        oldEnergy = defEn + fracEn
        # print("OLD ENERGY IS:", old_energy)

        ## Compute new energies, only stop if fracture or end of simulation
        stepsCounter = 0
        while (True):
            stepsCounter += 1
            newEnd = oldEnd + stepsCounter  ## Time step for new energy and crack path
            # if newEnd >= self.NBef+self.NAft:
            if newEnd >= stopCheckAt:
                return False, None, ()
            energies = {}
            ## Identifies all possible path cracks (easy in 1D)
            ## The doisplacement has already been simulated
            ## Computes all possible total energies for new paths and displacements
            # for i in range(1, floe.n):
            for i in range(1, 2):
                for newTuple in combinations(range(1, floe.n-2), i):        ## The two extreme springs must stay
                    newBrokenSprings = list(newTuple)
                    defEn = self.deformationEnergy(floeId, newBrokenSprings, newEnd)
                    fracEn = self.fractureEnergy(floeId, newBrokenSprings)
                    energies[newTuple] = defEn + fracEn

            # try:
            minConfig = sorted(energies.items(), key=lambda item: item[1])[0]
            # except IndexError:
            #     print("Got you !")

            ## Compare to the old energy and conclude
            if minConfig[1] < oldEnergy:
                print("\nFRACTURE OCCURRED FOR ICE FLOE " + str(floeId) + ':')

                print("     Starting configuration was:", (tuple(oldBrokenSprings), oldEnergy))
                print("     Minimum energy reached for:", minConfig)
                print("     Fracture happens at time step", newEnd, ", at time:", self.t[newEnd])

                return True, newEnd, minConfig

        else:
            print("     During the whole simulation, there was no fracture !")
            return False, None, ()


    def checkFracture(self, floeId):
        """
        Checks if fracture happens on any ice floe in the system.
        Returns the ids of the (potentially) two newly created floes.
        """
        fracture, fracPos, minConfig = self.griffithMinimization(floeId)

        if not fracture:
            self.checkFracFrom[floeId] = self.t.size  ## Will change later
            return False
        else:
            assert len(minConfig[0]) == 1, "Multiple spring fracturing simultanuously not yet studied "

            ## Checks that this is the lowest position before fractionning
            # lowest = min(self.checkFracFrom.values())
            # lowest = max(self.configurations.keys())
            # if fracPos < lowest:
            #     return False

            cS = minConfig[0][0]

            floe = self.floes[floeId]
            oldInitLengths = floe.init_lengths

            ## Split the nodes and the springs
            nodesL, nodesR = floe.nodes[:cS+1], floe.nodes[cS+1:]
            springsL, springsR = floe.springs[:cS], floe.springs[cS+1:]

            ## Set the old (left) floe's properties
            floe.nodes, floe.springs, floe.n = nodesL, springsL, len(nodesL)
            floe.init_lengths = oldInitLengths[:cS]

            ## Set new (right) floe's properties
            newFloe = deepcopy(floe)       ## <<------------- Implement this function !!
            newFloe.nodes, newFloe.springs, newFloe.n = nodesR, springsR, len(nodesR)
            newFloe.id = len(self.floes)
            for node in newFloe.nodes:
                node.parentFloe = newFloe.id
            newFloe.init_lengths = oldInitLengths[cS+1:]
            self.floes[len(self.floes)] = newFloe

            self.configurations[fracPos] = deepcopy(self.floes)
            ## Delete configurations that had been created before !
            toDel = [key for key in self.configurations.keys() if key > fracPos]
            for key in toDel:
                self.configurations.pop(key)

            ## If fracture, save each node's positions and speed
            for floe in self.floes.values():
                for node in floe.nodes:
                    node.x = self.x[fracPos, node.id]
                    node.vx = self.v[fracPos, node.id]

            ## Discard the positions and velocities after collision
            self.x = self.x[:fracPos + 1, :]
            self.v = self.v[:fracPos + 1, :]
            self.t = self.t[:fracPos + 1]

            self.checkFracFrom[floeId] = fracPos + 1
            self.checkFracFrom[newFloe.id] = fracPos + 1
            # self.checkFracFrom[floeId] = fracPos
            # self.checkFracFrom[newFloe.id] = fracPos

            return True


    def runSimulation(self):
        """
        Runs the simulation for the complete fracture problem
        """

        ## Run uniform mouvement phase up to the first collision
        self.computeBeforeContact()

        for key in self.floes.keys():
            self.checkFracFrom[key] = self.t.size

        while self.t.size <= self.NBef + self.NAft and max(self.collCount.values()) < 1000:

            self.computeAfterContact()

            ## (Potential) Fracture detection 
            floeDict = deepcopy(self.floes)
            for floe in floeDict.values():
                self.checkFracture(floe.id)

            ## Collision detection 
            for floe in self.floes.values():
                for node in floe.nodes:
                    res = self.checkCollision(node.id, node.rightNode)

            ## Assign the same value to all keys to check collision from now on
            smallestCheckCollFrom = min(self.checkCollFrom.values())
            for key in self.checkCollFrom.keys():
                self.checkCollFrom[key] = smallestCheckCollFrom
            self.collLoc.append(smallestCheckCollFrom)

            ## Assign the same value to all keys tp check fracture from now on
            smallestCheckFracFrom = min(self.checkFracFrom.values())
            for key in self.checkFracFrom.keys():
                self.checkFracFrom[key] = min([smallestCheckFracFrom, smallestCheckCollFrom])

        ## Truncate exeeding data (not strick;y necessary)  
        self.x = self.x[:self.NBef+self.NAft+2, :]
        self.v = self.v[:self.NBef+self.NAft+2, :]
        self.t = self.t[:self.NBef+self.NAft+2]


        print("FINISHED")


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

        colors = ['b', 'g', 'r', 'c', 'm', 'y']

        # ax.set_xlim(min_X, max_X)
        # ax.set_ylim(-4 * max_R, 4 * max_R)
        # ax.set_aspect('equal', adjustable='box')

        dt = self.tBef / self.NBef
        di = int(1 / fps / dt)
        if di == 0:
            print("Error: The frame rate is to high. Reduce it please !")
            exit(1)

        img_list = []

        print("Generating frames ...")

        ## For loop to update the floes nodes, then plot
        configKeys = list(self.configurations.keys())
        for j in range(0, len(configKeys)):
            floes = self.configurations[configKeys[j]]

            start = configKeys[j]
            end = configKeys[j+1] if j != len(configKeys)-1 else self.t.size
            for i in range(start, end, di):

                if i >= self.t.size: break

                print("  ", (i // di), '/', self.t.size // di)

                ax.set_title("t = "+str(np.round(self.t[i], 2)), size="xx-large")

                for floe in floes.values():
                    for node in floe.nodes:
                        node.x = self.x[i, node.id]
                        node.vx = self.v[i, node.id]

                    floe.plot(figax=(fig,ax), color=colors[floe.id%len(colors)])

                ax.set_xlim(min_X, max_X)
                ax.set_ylim(-2 * max_R, 2 * max_R)
                ax.set_aspect('equal', adjustable='box')

                img_list.append(fig2img(fig))           ### Use tight borders !!!!!!!!!!!!!!!!!!!!!!!

                plt.cla()      # Clear the Axes ready for the next image.

        imageio.mimwrite(filename, img_list)
        print("OK! saved file '"+filename+"'")

        if openFile:
            ## Open animation
            os.system('gthumb '+filename)     ## Only on Linux



    def plot_energy(self, figax):
        """
        Plots the total energy of the system before and after first chocs
        """
        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        ## Masses des noeuds
        m = np.zeros((self.nbNodes))
        for floe in self.floes.values():
            for node in floe.nodes:
                m[node.id] = floe.m

        ## Energie cinetique
        E_c = (0.5 * np.sum(m[np.newaxis, :] * self.v**2, axis=-1))

        ## Energie elastique
        E_el = np.zeros_like(self.t)
        initLen = np.zeros((self.nbNodes - 1))
        for start, floes in self.configurations.items():
            k = np.zeros((self.nbNodes - 1))

            for floe in floes.values():
                for spring in floe.springs:
                    k[spring.id] = floe.k
                    initLen[spring.id] = spring.L0

            E_el[start:] = 0.5 * np.sum(k[np.newaxis, :] * (self.x[start:, 1:] - self.x[start:, :-1]
                                                            - initLen[np.newaxis, :])**2, axis=-1)

        ## Energie dissipative
        E_r = np.zeros_like(self.t)
        for start, floes in self.configurations.items():
            mu = np.zeros((self.nbNodes - 1))

            for floe in floes.values():
                for spring in floe.springs:
                    mu[spring.id] = floe.mu

            integrand = np.sum(mu[np.newaxis, :] * (self.v[start:, 1:] - self.v[start:, :-1]) ** 2, axis=-1)
            t = self.t[start+1:] - self.t[start:-1]
            # E_r[start:-1] = E_r[start-1] + np.cumsum(integrand[:-1] * t)
            E_r[start:-1] = E_r[start-1] + np.cumsum(integrand[:-1] * t)

        E_r[-1] = np.nan

        E_tot = E_c + E_el + E_r

        # print("Énergie totale immediatement avant 1er choc:", E_tot[N_first])
        # print("Énergie totale immediatement après 1er choc:", E_tot[N_first + 1])
        # print("Rapport APRÈS/AVANT:", E_ap[N_first + 1] / E_av[N_first])
        # print("Epsilon:", self.eps)

        ax.plot(self.t, E_tot, "-", linewidth=2, label="énergie totale")

        ax.plot(self.t, E_c, "--", linewidth=1, label="énergie cinétique")
        ax.plot(self.t, E_el, "--", linewidth=1, label="énergie élastique")
        ax.plot(self.t, E_r, "--", linewidth=1, label="énergie dissipée")

        # for i, N_choc in enumerate(self.contact_indices):
        #     # label = "1er" if i==0 else str(i+1)+"eme"
        #     # ax.plot([self.t[N_choc+1]], [E_ap[N_choc+1]], marker='X', label=label+" choc")
        #     label = "chocs" if i==0 else None
        #     ax.plot([self.t[N_choc+1]], [E_ap[N_choc+1]], 'kX', alpha=0.5, label=label)

        collLoc = np.unique(self.collLoc)
        for i in range(0, len(collLoc)):
            label = "collisions" if i == 0 else None
            if collLoc[i] < self.t.size:
                ax.plot([self.t[collLoc[i]]], [E_tot[collLoc[i]]], 'k.', label=label, alpha=0.5)

        fracLoc = list(self.configurations.keys())
        for i in range(1, len(fracLoc)):
            label = "fractures" if i == 1 else None
            ax.plot([self.t[fracLoc[i]]], [E_tot[fracLoc[i]]], 'rX', label=label)

        ax.set_title("Énergie mécanique totale")
        ax.set_xlabel("temps")
        text = 'rapport fin/début: ' + str(np.round(E_tot[-2] / E_tot[0], 3)) \
               + '\nepsilon: ' + str(np.round(self.eps, 3))
        ax.text(0.9, 0.1, text,
             horizontalalignment='right',
             verticalalignment='baseline',
             transform=ax.transAxes)
        ax.legend()
        fig.tight_layout()

        return figax



    def plot_momentum(self, figax):
        """
        Plots the momentum of the system before and after first choc
        """
        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        ## Masses des noeuds
        m = np.zeros((self.nbNodes))
        for floe in self.floes.values():
            for node in floe.nodes:
                m[node.id] = floe.m

        P = np.sum(m * self.v, axis=-1)

        ax.set_ylim([0, 2*P[0]])

        ax.plot(self.t, P, label="quantité de mouvement")
        # for i, N_choc in enumerate(self.contact_indices[:]):
        #     label = "chocs" if i==0 else None
        #     ax.plot([self.t[N_choc+1]], [P_ap[N_choc+1]], 'kX', alpha=0.5, label=label)

        ax.set_title("Quantité de mouvement")
        ax.set_xlabel("temps")
        text = 'rapport fin/début: ' + str(np.round(P[-1] / P[0], 2)) \
               + '\nepsilon: ' + str(self.eps)
        ax.text(0.9, 0.1, text,
             horizontalalignment='right',
             verticalalignment='baseline',
             transform=ax.transAxes)
        ax.legend()
        fig.tight_layout()

        return (fig, ax)



    def plot_positions(self, node_ids=None, figax=None):
        """
        Plot the positions of (some) nodes of an ice floe part of this percussion problem.
        """

        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        if not node_ids:
            node_ids = list(range(self.nbNodes))

        try:
            for i in node_ids:
                ax.plot(self.t, self.x[:, i], label=r"$z_"+str(i)+"$")
        except IndexError:
            print("Error plotting: A given node id not valid!")

        ax.set_title("Trajectoires des noeuds")
        ax.set_xlabel("temps")
        if len(node_ids) < 8:
            ax.legend()
        fig.tight_layout()

        return (fig, ax)


    def plot_velocities(self, node_ids=None, figax=None):
        """
        Plot the positions of (some) nodes of an ice floe part of this percussion problem.
        """

        if figax:
            fig, ax = figax
        else:
            figax = plt.subplots()
            fig, ax = figax

        if not node_ids:
            node_ids = list(range(self.nbNodes))

        try:
            for i in node_ids:
                ax.plot(self.t, self.v[:, i], label=r"$z_"+str(i)+"$")
        except IndexError:
            print("Error plotting: A given node id not valid!")

        ax.set_title("Vitesses des noeuds")
        ax.set_xlabel("temps")
        if len(node_ids) < 8:
            ax.legend()
        fig.tight_layout()

        return (fig, ax)