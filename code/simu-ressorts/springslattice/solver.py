import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from pathlib import Path

from . import mesh
from .cython_loop import jacobian_loop

np.seterr(all="print")
#np.seterr(under="ignore")

def euclidian(a, b):
  return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class SpringNetwork:
  def __init__(self, intensity, total_mass, stiffness, viscosity=0, impact_velocity=0.1, impact_mass=10, regularity=(0,1)):
    self.intensity = intensity
    self.total_mass = total_mass
    self.k = stiffness
    self.v = viscosity
    self.impact_velocity = impact_velocity
    self.impact_mass = impact_mass

    self.mesh = mesh.Mesh(intensity, regularity)
    self.n = len(self.mesh.nodes) # Number of particles
    self.m = total_mass/self.n        # Mass of a particle
    
    # Adjacency matrix and Lengh at equilibrium matrix (resp C -- in sparse row/col format -- and L0)
    self.row = []
    self.col = []
    self.L0 = []
    for i, n1 in enumerate(self.mesh.nodes):
      for n2 in n1.neighbors:
        j = n2.id_number
        if j < i:
          self.row += [i,j]
          self.col += [j,i]
          d = self.mesh.d_eq(n1, n2)
          self.L0 += [d, d]
    
    self.Idx = np.array([n.x for n in self.mesh.nodes])
    self.Idy = np.array([n.y for n in self.mesh.nodes])
    self.center_mass = np.array([sum([p.x for p in self.mesh.nodes])/self.n, sum([p.y for p in self.mesh.nodes])/self.n])
    
    # Choosing impact node and fix node
    self.impact_index = np.argmin([n.x for n in self.mesh.nodes])
    self.impact_node = self.mesh.nodes[self.impact_index]
    self.fix_indexes = [np.argmax([n.x for n in self.mesh.nodes])]
    neighbors = self.mesh.boundary_mesh.neighbors(self.mesh.nodes[self.fix_indexes[0]])
    other_fix_node = max(neighbors, key= lambda p: p.x)
    self.fix_indexes.append(other_fix_node.id_number)
    self.fix_indexes.sort()
    self.fix_nodes = [self.mesh.nodes[i] for i in self.fix_indexes]
    self.reduced_index = lambda x: x if x < self.fix_indexes[0] else (x-1 if x < self.fix_indexes[1] else x-2)
    
    self.gen_init_data()
    
  def gen_init_data(self):
    self.mass_matrix = np.full((2*self.n), self.m)
    self.mass_matrix[self.impact_index] += self.impact_mass
    self.mass_matrix[self.impact_index+self.n] += self.impact_mass
    
    # Percussion : initial values
    dY0 = np.zeros((2*self.n))
    dY0[self.impact_index], dY0[self.impact_index + self.n] = self.impact_velocity*(self.center_mass - self.impact_node.to_array())
    
    self.CEQ = np.hstack((np.zeros((2*self.n)), self.Idx, self.Idy))    
    self.CI = np.hstack((dY0, self.Idx, self.Idy))
  
  def reset_constants(self, m, k, v, impact_velocity, impact_mass):
    self.total_mass, self.k, self.v = m, k, v
    self.m = m/self.n
    self.impact_velocity, self.impact_mass = impact_velocity, impact_mass
    self.gen_init_data()
    
  def test_explicit_jacobian(self):
    Err = 10**-5
    self.Jac2 = self.numerical_Jacobian()
    if scipy.linalg.norm(self.Jacobian(self.CEQ)-self.Jac2) > Err:
      raise RuntimeError

  def F(self, t, Z):
    """
    Dummy function
    """
    return self.Fhom(Z)
    
  def Fhom(self, Z):
    """
    Define F such that Z' = F(Z).
    We recall that : Z = (dq, q) with q the position, dq the speed.
    """
    n, k, m, v = self.n, self.k, self.mass_matrix, self.v
    L0 = self.L0
    row, col = self.row, self.col
    
    qx, qy =  Z[2*n:3*n], Z[3*n:4*n]
    dq = Z[0:2*n]
    dqx, dqy = dq[0:n], dq[n:2*n]

    data_L_L0_ux = []
    data_L_L0_uy = []
    for k1, (i,j) in enumerate(zip(row, col)):
      dist = euclidian((qx[i], qy[i]), (qx[j], qy[j]))
      dLdt = ((qx[i]-qx[j])*(dqx[i] - dqx[j]) + (qy[i] - qy[j])*(dqy[i] - dqy[j])) / dist
      data_L_L0_ux += [- k / m[i] * (dist - L0[k1]) * (qx[i] - qx[j]) / dist \
               - v / m[i] * dLdt * (qx[i] - qx[j]) / dist]
      data_L_L0_uy += [- k / m[i] * (dist - L0[k1]) * (qy[i] - qy[j]) / dist \
               - v / m[i] * dLdt * (qy[i] - qy[j]) / dist]
    L_L0_ux = sparse.coo_matrix((data_L_L0_ux, (row, col)), shape=(n, n)).tocsc()
    L_L0_uy = sparse.coo_matrix((data_L_L0_uy, (row, col)), shape=(n, n)).tocsc()
  
    dp = np.hstack([L_L0_ux.sum(axis=1).getA1(), L_L0_uy.sum(axis=1).getA1()])

    # Fix point trick
    for i in self.fix_indexes:
      dp[i], dp[i+n] = 0, 0
      dq[i], dq[i+n] = 0, 0
    

    # 2nd order -> 1rst order
    return np.hstack((dp, dq))

  def numerical_Jacobian(self):
    import numdifftools as nd
    return nd.Jacobian(self.Fhom)(self.CEQ)

  def Jacobian(self, Z, reduced=False):
    return self._Jacobian(Z, reduced, with_inverse=False)[0]

  def Jacobian_with_inverse(self, Z, reduced=False):
    return self._Jacobian(Z, reduced, with_inverse=True)
  
  def _Jacobian(self, Z, reduced=False, with_inverse=False):
    """
    Compute Jacobian matrix of F at point Z.
    We recall that :
    Z' = F(Z), with Z = (dq, q), with q the position, dq the speed.
    """
    n, k, m, v = self.n, self.k, self.mass_matrix, self.v
    L0 = self.L0
    dqx, dqy, qx, qy = Z[0:n], Z[n:2*n], Z[2*n:3*n], Z[3*n:4*n]

    row, col, L0 = list(zip(*filter(lambda x: x[0] not in self.fix_indexes, zip(self.row, self.col, L0))))
    L0 = np.array(L0)
    return jacobian_loop(n, k, m, v, L0, row, col, qx, qy, dqx, dqy, reduced, self.reduced_index, self.fix_indexes, self.impact_index, with_inverse, with_cholesky=False)

  def linearizeF(self):
    Z0 = self.CEQ
    Jac = self.Jac
    FZ0 = self.F(0, Z0)
    def Flin(t, Z):
      return FZ0 + Jac.dot(Z-Z0)
    return Flin

  def compute_sol_lin(self, t_span, *args, **kargs):
    Flin = self.linearizeF()
    self.sol_lin = solve_ivp(Flin, t_span, self.CI, *args, **kargs)
    return self.sol_lin
  
  def compute_sol(self, t_span, *args, **kargs):
    self.sol = solve_ivp(self.F, t_span, self.CI, *args, **kargs)
    return self.sol

  def plot_mesh(self, save_file=None):
    fig, ax = self.mesh.plot(save_file=save_file)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax.plot(self.impact_node.x, self.impact_node.y, 'gs')
    for p in self.fix_nodes:
      ax.plot(p.x, p.y, 'ys')

    if save_file:
      p = Path(save_file)
      if p.exists():
        p.unlink()       
      fig.savefig(save_file)
      plt.close(fig)
    
    else:      
      plt.show()
  
  def plot_solution(self, sol, save_file=False):
    Idx, Idy = self.Idx, self.Idy
    n = self.n
    fig, ax= plt.subplots()

    ax.clear()
    ax.axis([0, 1, 0, 1])
    ax.axis('off')
    qver = ax.quiver(Idx, Idy, sol.y[2*n:3*n, 0] - Idx, sol.y[3*n:4*n,0] - Idy, angles='xy', scale=1, scale_units='xy')
    text = ax.title
  
    def animate(i):
      qver.set_UVC(sol.y[2*n:3*n, i] - Idx, sol.y[3*n:4*n, i] - Idy)
      text.set_text(r'frame {} at time {}'.format(i, np.round(sol.t[i], decimals=1)))
      return qver, text
    
    self.ani = FuncAnimation(fig, animate, range(len(sol.t)), interval=100, blit=True)

    if save_file:
      Writer = animation.writers['ffmpeg']
      writer = Writer(fps=40, bitrate=1800)
      p = Path(save_file)
      if p.exists():
        p.unlink()       
      self.ani.save(save_file, writer=writer)
      plt.close(fig)
    else:      
      plt.show()
      return fig

  def plot_eig(self, save_pictures=False):
    eigva, eigve = scipy.linalg.eig(self.Jacobian(self.CEQ).todense())
    eigv = list(zip(eigva, eigve))
    eigv.sort(key=lambda eg: abs(eg[0]))
    eigva, eigve = list(zip(*eigv))
    
    Idx, Idy = self.Idx, self.Idy
    n = self.n
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    eigva_complex = [(e.real, e.imag) for e in eigva]
    eigva_plot = [ax1.plot(p[0], p[1], 'kx')[0] for p in eigva_complex]
    lambdamax = np.amax(list(map(np.absolute, eigva)))
    ax1.set_title('Eigenvalue {} : {}'.format(0, eigva[0]))

    ax2.axis([0, 1, 0, 1])
    size = [[100*euclidian((eigve[j][i], eigve[j+n][i]), (0, 0)) for j in range(n)] for i in range(len(eigva))]
    eigve_plot = ax2.scatter(Idx, Idy, color='r', s=size[:][0])

    def animate(i):
      if i == 0:
        pred = len(eigva)-1
      else:
        pred = i-1
      eigva_plot[i].set_color('red')
      eigva_plot[i].set_marker('o')
      eigva_plot[pred].set_color('black')
      eigva_plot[pred].set_marker('x')
      ax1.set_title('Eigenvalue {0} : {1.real:f} + {1.imag:f}j'.format(i, eigva[i]))
      eigve_plot.set_sizes(size[:][i])
      return tuple(eigva_plot) + (eigve_plot,)
    
    if save_pictures:
      path = Path(save_pictures)
      path.mkdir()
      for i in range(len(eigva)):
        animate(i)
        fig.savefig(save_pictures + '/' + str(i) + '.png')
  
  def eigjump(self):
    jac, jac_inv = self.Jacobian_with_inverse(self.CEQ, reduced=True)
    eigva = list(sparse.linalg.eigs(jac, OPinv=jac_inv, k=5, which='LM', sigma=0, tol=0.01, return_eigenvectors=False))
    eigva.sort(key=lambda x: x.real, reverse=True)
    return abs(eigva[4].real)/abs(eigva[3].real)
  
  def eigjump_full(self):
    eigva = list(scipy.linalg.eigvals(self.Jacobian(self.CEQ).todense()))
    eigva.sort(key=lambda x: x.real, reverse=True)
    return abs(eigva[12].real)/abs(eigva[11].real)
