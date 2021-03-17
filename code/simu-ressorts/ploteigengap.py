import argparse
import numpy as np
from math import log
import matplotlib.pyplot as plt

from springslattice import solver

def jump(sn, epsilon):
  sn.reset_constants(sn.total_mass, sn.k/epsilon**2, sn.v/epsilon, sn.impact_velocity, sn.impact_mass/epsilon**2)
  jump = sn.eigjump()
  sn.reset_constants(sn.total_mass, sn.k*epsilon**2, sn.v*epsilon, sn.impact_velocity, sn.impact_mass*epsilon**2)
  return jump


class SpringGenerator:
  def __init__(self, mean_number_part, mass, stiffness, viscosity, impact_velocity, impact_mass, regularity):
    self.mean_number_part, self.mass, self.stiffness, self.viscosity, self.impact_velocity, self.impact_mass = mean_number_part, mass, stiffness, viscosity, impact_velocity, impact_mass
    self.regularity = regularity
  
  def generate(self):
    sn = solver.SpringNetwork(self.mean_number_part, self.mass, self.stiffness, self.viscosity, self.impact_velocity, self.impact_mass, self.regularity)    
    #print(sn.mesh.regularity)

    while False:
      sn = solver.SpringNetwork(args.mean_number_part, args.mass, args.stiffness, args.viscosity, args.impact_velocity, args.impact_mass)
      if sn.impact_node.x < 0.65 and sn.impact_node.x > 0.35:
        break
    
    while False:
      sn = solver.SpringNetwork(self.mean_number_part, self.mass, self.stiffness, self.viscosity, self.impact_velocity, self.impact_mass)
      #if sn.n >= 0.95*sn.intensity and sn.n <= 1.05*sn.intensity:
      if sn.n == sn.intensity:
        break
    return sn

def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Simulates the behavior of a Poisson-Delaunay mass-spring lattice')
  parser.add_argument('mode', action="store", type=int, help="Mode (see script for details)")
  parser.add_argument('-n', '--mean_number_part', type=float, default='15', help="Mean number of particles for the Poisson-Delaunay process")
  parser.add_argument('-d', '--draws', type=int, default='3', help="Number of draws")
  parser.add_argument('-m', '--mass', default='1', type=float, help="Total mass of the spring lattice")
  parser.add_argument('-im', '--impact_mass', default='1', type=float, help="Mass of the impact particle")
  parser.add_argument('-k', '--stiffness', default='1', type=float, help="Stiffness of springs")
  parser.add_argument('-v', '--viscosity', default='1', type=float, help="Viscosity coefficient")
  parser.add_argument('-iv', '--impact_velocity', default='1', type=float, help="Velocity of colliding object")
  parser.add_argument('-rmin', '--regularity-min', default='0', type=float, help="Minimum regularity of Poisson-Delaunay lattice")
  parser.add_argument('-rmax', '--regularity-max', default='1', type=float, help="Minimum regularity of Poisson-Delaunay lattice")
  parser.add_argument('-bins', '--bins', default='40', type=int, help="Bins for the histogram")
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-w', '--write', action='store_true', help='Writes result in gap_results/.txt file')
  group.add_argument('-r', '--read', action='store_true', help='Reads results in gap_results/.txt file')
  group.add_argument('-p', '--plot', action='store_true', help='Plots computed results')


  
  args = parser.parse_args(raw_args)
  
  spg = SpringGenerator(args.mean_number_part, args.mass, args.stiffness, args.viscosity, args.impact_velocity, args.impact_mass, (args.regularity_min, args.regularity_max))
  fig, ax = plt.subplots()
  
  assert args.regularity_min < args.regularity_max
  
  if args.mode == 1:
    exp = np.arange(2, 5.0, 1)
    eps = [10**-e for e in exp]
    leps = [log(e) for e in eps]
    for _ in range(args.draws):
      sn = spg.generate()
      eigjumps = [log(jump(sn, ep)) for ep in eps]
      ax.plot(leps, eigjumps)
    plt.show()

  elif args.mode == 2:
    exp = np.arange(2, 5, 1)
    eps = [10**-e for e in exp]
    leps = [log(e) for e in eps]
    for n in [10, 50, 100, 200, 500, 1000, 2000]:
      spg.mean_number_part = n
      sn = spg.generate()
      eigjumps = [log(jump(sn, ep)) for ep in eps]
      ax.plot(leps, eigjumps)
    plt.show()

  elif args.mode == 3:
    regularity = [spg.generate().mesh.regularity for _ in range(args.draws)]
    ax.hist(regularity, bins=20)
    plt.show()

  elif args.mode == 4:
    if args.read:
      with open("gap_results/gap_{}.txt".format(int(args.mean_number_part)), 'r') as data:
        eigjumps_a0 = [float(d) for d in data.read().split()]
        ax.hist(eigjumps_a0, bins=args.bins, range=(0, 3))
        plt.show()
    elif args.write:
      eps = 10**(-4)
      with open("gap_results/gap_{}.txt".format(int(args.mean_number_part)), 'a', buffering=1) as data:
        for _ in range(args.draws):
          data.write("{}\n".format(log(jump(spg.generate(), eps)) + 2*log(eps)))
    else:
      eps = 10**(-4)
      eigjumps_a0 = [log(jump(spg.generate(), eps)) + 2*log(eps)for _ in range(args.draws)]
      ax.hist(eigjumps_a0, bins=args.bins, range=(0, 3))
      if args.plot:
        plt.show()

  elif args.mode == 5:
    eigjumps_alpha = []
    ep1 = 10**(-4)
    ep2 = 10**(-5)
    for _ in range(args.draws):
      sn = spg.generate()
      eigjumps_alpha.append((log(jump(sn, ep2)) - log(jump(sn, ep1)))/(log(ep2/ep1)))
    print(eigjumps_alpha)
  
  elif args.mode == 6:
    import random
    random.seed(1)
    for _ in range(args.draws):
      sn = spg.generate()
      eigjump = log(jump(sn, 1))
      if eigjump < 0.4:
        sn.plot_mesh("gap_results/small_gap/{}.png".format(random.randint(1, 100000)))
      elif eigjump > 2.2:
        sn.plot_mesh("gap_results/large_gap/{}.png".format(random.randint(1, 100000)))
    
if __name__ == '__main__':
  run()

    
