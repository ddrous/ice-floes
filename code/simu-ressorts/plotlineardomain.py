import matplotlib
import math
matplotlib.use('Agg')
from matplotlib import colors
from scipy.integrate import solve_ivp
import numpy as np
import itertools
import matplotlib.pyplot as plt
import h5py
import argparse
import collections
import time

from springslattice.solver import SpringNetwork

def norme_sol(sol, sol2):
  return np.linalg.norm(sol.y-sol2.y)

def min_to_1(e):
  return min(1, e)

def compute():
  initial_speed = np.linspace(0.01, 0.1, 10)
  stiffnesses = np.linspace(0.1, 10000, 10)
  mean_number = 50
  total_mass = 10
  viscosity = 0.01
  impact_mass = 10
  tf = 2
  sn = SpringNetwork(mean_number, total_mass, 1, viscosity, 1, impact_mass)
  number_particles = sn.n
  result = []
  
  for vp, k in itertools.product(initial_speed, stiffnesses):
    sn.reset_constants(total_mass, k, viscosity, vp)
    F = sn.F
    sol = solve_ivp(F, (0, tf), sn.CI)
    Flin = sn.linearizeF()
    sol_lin = solve_ivp(Flin, (0, tf), sn.CI, t_eval=sol.t)
    result.append(norme_sol(sol, sol_lin))
  
  with h5py.File('data.hdf5', 'a') as data:
    try:
      m = max(map(int, data))
    except ValueError:
      m = 0
    grp = data.create_group('{}'.format(m+1))
    problem_dataset = grp.create_dataset('InitValues', (5,)) # Values of Mean_number, n, m, r, tf
    problem_dataset[:] = [mean_number, number_particles, total_mass, viscosity, tf]
    mesh = grp.create_group('mesh')
    sn.mesh.save_hdf5(mesh)
    tuple_speed_stiffness = list(itertools.product(initial_speed, stiffnesses))
    result_dataset = grp.create_dataset('Results', (len(tuple_speed_stiffness),3), compression='gzip') #3 row : initial_speed, stifness and result
    result_dataset[:,0:2] = tuple_speed_stiffness
    result_dataset[:,2] = result
    data.flush()


def plot(group, log_colored=False):
  with h5py.File('data.hdf5', 'r') as data:
    grp = data.get('{}'.format(group))
    fig, ax = plt.subplots()
    vp, k, results = zip(*grp.get('Results'))
    if log_colored:    
      min_result, max_result = min(results), max(results)
      size_vp = collections.Counter(k)[k[0]]
      size_k = collections.Counter(vp)[vp[0]]
      vp = np.array(vp).reshape((size_vp, size_k))[:,0] # We make results with itertools. The order is known. k varies first, then vp.
      k = k[0:size_k]
      results = np.transpose(np.array(results).reshape((size_vp, size_k)))
      pcm = ax.pcolormesh(vp, k, results, norm=colors.LogNorm(vmin=min_result, vmax=max_result), cmap='PuBu_r')
      fig.colorbar(pcm, ax=ax, extend='max')
    else:
      results = map(min_to_1, results)
      zeros = np.zeros((len(vp)))
      rgba_colors = list(zip(*(zeros,)*3, results))
      ax.scatter(vp, k, c=rgba_colors)
    ax.set_xlabel('Celerity of percussion')
    ax.set_ylabel('Stiffness')
  plt.savefig('Linear domain {}'.format(group))


def list_groups():
  with h5py.File('data.hdf5', 'r') as data:
    for g in data:
      print(g)

def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Computes and plots error between linear and non-linear simulation of a system of springs')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-c', '--compute', dest='compute', action='store_true', help='Computes and stores the result in data.hdf5')
  group.add_argument('-p', '--plot', type=int, dest='plot', help='Plot the PLOT-th result from data.hdf5') #, action='store_true')
  group.add_argument('-l', '--list', dest='list_groups', action='store_true', help='Lists simulations that are in data.hdf5')
  parser.add_argument('-lc', '--logcolored', dest='log_colored', action='store_true', help='If set, plots use a log-scale for color')
  args = parser.parse_args()

  begin = time.time()
  if args.compute:
    compute()
  elif args.plot:
    plot(args.plot, args.log_colored)
  elif args.list_groups:
    list_groups()

  end = time.time()

  print('Done in {} seconds.'.format(end-begin))

if __name__ == '__main__':
  run()
