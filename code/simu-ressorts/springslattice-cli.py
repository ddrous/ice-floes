import argparse
from scipy.integrate import solve_ivp
import numpy as np
from enum import Enum

from springslattice import solver, symplectic

class Solver(Enum):
    euler = 'euler'
    symplectic = 'symplectic'

    def __str__(self):
        return self.value

def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Simulates the behavior of a Poisson-Delaunay mass-spring lattice')
  parser.add_argument('-n', '--mean_number_part', type=float, default='15', help="Mean number of particles for the Poisson-Delaunay process")
  parser.add_argument('-m', '--mass', default='10', type=float, help="Total mass of the spring lattice")
  parser.add_argument('-im', '--impact_mass', default='10', type=float, help="Mass of the impact particle")
  parser.add_argument('-k', '--stiffness', default='10', type=float, help="Stiffness of springs")
  parser.add_argument('-v', '--viscosity', default='1', type=float, help="Viscosity coefficient")
  parser.add_argument('-iv', '--impact_velocity', default='0.1', type=float, help="Velocity of colliding object")
  parser.add_argument('-t', '--final_time', default='10', type=float, help="Final time of simulation")
  parser.add_argument('-s', '--solver', default='euler', type=Solver, choices=list(Solver), help="Integration method")
  parser.add_argument('-l', '--linearize', action="store_true", help="Linearize system")
  parser.add_argument('-e', '--eigenvalues', action="store_true", help="Plots eigenvalues")

  args = parser.parse_args(raw_args)
  
  sn = solver.SpringNetwork(args.mean_number_part, args.mass, args.stiffness, args.viscosity, args.impact_velocity, args.impact_mass)
  CI = sn.CI

  if args.linearize:
    F = sn.linearizeF()
  else:
    F = sn.F
  
  if args.final_time > 0:
    if args.solver==Solver.euler:
      sol = solve_ivp(F, (0, args.final_time), sn.CI)
    
    elif args.solver==Solver.symplectic:
      sol = solve_ivp(F, (0, args.final_time), sn.CI, method=symplectic.SIE, h_abs=0.02)
    
    sn.plot_solution(sol)

  if args.eigenvalues:
    sn.plot_eig()
  

if __name__ == '__main__':
  run()

