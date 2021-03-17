from multiprocessing import Process, Queue
from queue import Empty as EmptyQueue
import warnings
import numpy as np
import scipy
import scipy.integrate
import itertools
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from math import sqrt

from . import stiff_matrix as stiff
from . import mesh as msh
from . import finite_elements as el
from . import geometry as geo
from . import fracture_iterators
from . import problem_data
from . import scalar_product

##################
# Numerical Errors
##################
try:
  from scikits.umfpack import UmfpackWarning
  warnings.filterwarnings('error', category=UmfpackWarning)
except ImportError:
  class UmfpackWarning(Exception):
    pass

warnings.filterwarnings('ignore', category=scipy.integrate.IntegrationWarning)
ENERGY_TOLERENCE = 10e-5
ENERGY_PRECISION = 1

class NumericalError(RuntimeError):
  """
  Exception raised when the solution given by the solver does not solve the system.
  """
  pass


###################
# Solutions classes
###################
class Infinite_Energy:
  """
  Return value if error in Partial_Imposed_Fracture_Solution
  """
  energy = np.inf
  elastic_energy = np.inf
  fracture_energy = np.inf

  @classmethod
  def strip_to_lazy_solution(cls):
    return cls
  
  @classmethod
  def to_full(cls, *args):
    return cls


class Solution:
  def __init__(self, physical_data, field, time):
    self.physical_data = physical_data
    self.field = field
    self._time = time
    
    self.stiff_matrix, self._sparse_stiff_data = stiff.compute_interior_stiff_matrix(self.field, self.physical_data.stiffness_tensor)
    self.boundary_matrix = stiff.compute_boundary_matrix(self.field, self.physical_data.stiffness_tensor)
    self.boundary_stiff = stiff.compute_boundary_stiff(self.field, self.physical_data.stiffness_tensor)
    self.u_bound = np.array(list(map(lambda p: physical_data.boundary_displacement(time, p), [e.base_node for e, e_ in zip(*[iter(self.field.boundary_elements)]*2)]))).flatten()
    self._solve()

  def _solve(self):
    """
    The vector field u is the displacement.
    """
    self.u_int = scipy.sparse.linalg.spsolve(self.stiff_matrix, -self.boundary_matrix.dot(self.u_bound))
    self.u = np.hstack([self.u_int, self.u_bound])
    self.elastic_energy = self.stiff_matrix.dot(self.u_int).dot(self.u_int) \
                  + 2*self.boundary_matrix.dot(self.u_bound).dot(self.u_int) \
                  + self.boundary_stiff.dot(self.u_bound).dot(self.u_bound)
    
    if np.max(np.abs(self.stiff_matrix.dot(self.u_int) + self.boundary_matrix.dot(self.u_bound))) > ENERGY_PRECISION:
      raise NumericalError
    elif self.elastic_energy <= -ENERGY_TOLERENCE:
      raise NumericalError
    if self.elastic_energy <= ENERGY_TOLERENCE:
      self.elastic_energy = 0
    try:
      # Integration of non continuous function, might be wrong
      self.fracture_energy, _ = scipy.integrate.quad(lambda x: self.physical_data.toughness_field(self.fracture.curvilinear_abscissa(x)), 0, self.fracture.lengh)
    except AttributeError:
      self.fracture_energy = 0
    self.energy = self.elastic_energy + self.fracture_energy

  def change_time(self, time):
    self.u_bound = np.array(list(map(lambda p: self.physical_data.boundary_displacement(time, p), [e.base_node for e, e_ in zip(*[iter(self.field.boundary_elements)]*2)]))).flatten()
    if isinstance(self.physical_data.boundary_displacement, problem_data.Linear_Displacement):
      t1, t2 = self._time, time
      self.elastic_energy = (t2/t1)**2*self.elastic_energy
      self.energy = self.elastic_energy + self.fracture_energy
      self.u_int = (t2/t1)*self.u_int
      self.u = np.hstack([self.u_int, self.u_bound])
    else: 
      self._solve()
    self._time = time

  def plot(self, figax=None):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    ax.set_aspect('equal')
    self.mesh.plot(figax)
    self.physical_data.toughness_field.plot(figax)
    return figax

  @property
  def nbr_segments(self):
    try:
      nbr_segments = len(self.fracture.segments)
    except AttributeError:
      nbr_segments = 0
    return nbr_segments

  def strip_to_lazy_solution(self):
    return Lazy_Solution(self)

  def to_full(self, *args):
    return self


class Lazy_Solution:
  def __init__(self, full_solution):
    self.energy = full_solution.energy
    self._time = full_solution._time
    self.elastic_energy = full_solution.elastic_energy
    self.fracture_energy = full_solution.fracture_energy
    self.fracture = msh.Lazy_Fracture(full_solution.fracture)
    self.physical_data = full_solution.physical_data
    self._tip_enrichement = full_solution._tip_enrichement

  def to_full(self, mesh, physical_data):
    if self.fracture is None:
      return Classical_Solution(mesh, physical_data)
    else:
      fracture = self.fracture.to_full(mesh)
      return Imposed_Fracture_Solution(mesh, physical_data, fracture, self._time, self._tip_enrichement)

  def change_time(self, time):
    assert isinstance(self.physical_data.boundary_displacement, problem_data.Linear_Displacement)
    t1, t2 = self._time, time
    self.elastic_energy = (t2/t1)**2*self.elastic_energy
    self.energy = self.elastic_energy + self.fracture_energy
    self._time = time


class Classical_Solution(Solution):
  """
  Computes the displacement field and energy of the classical (i.e. purely elastic) solution.
  """
  def __init__ (self, mesh, physical_data, time=1):
    self.mesh = mesh
    self.field = el.Finite_Elements(mesh)
    self.fracture = msh.No_Fracture()
    super().__init__(physical_data, self.field, time)

  def plot_displacement(self, figax=None):
    interior_elements = self.field.interior_elements
    boundary_elements = self.field.boundary_elements
    
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    
    affine = []
    for i, element in enumerate(interior_elements + boundary_elements):
      if element.vector == "u_y":
        continue
      node = element.base_node
      u, v = self.u[i], self.u[i+1]
      affine.append((node.x, node.y, u, v))
    
    #ax.figure(solution_id)
    X, Y, U, V = tuple(zip(*affine))
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy')

    return figax 
  
  def plot_energy(self, figax=None):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    
    E = []
    for t in self.mesh.triangles:
      elements = [e for n in t.nodes for e in self.field.elements_on_node[n]]
      coefficients = [self.u_int[self.field.interior_elements.index(e)] if e in self.field.interior_elements else self.u_bound[self.field.boundary_elements.index(e)] for e in elements]
      energy = 0
      for e1, c1, in zip(elements, coefficients):
        for e2, c2 in zip(elements, coefficients):
          energy += c1*c2*scalar_product.scalar_product_over_triangle(t, self.physical_data.stiffness_tensor, e1, e2)
      E.append(energy/t.area)
    
    scalar_map = mplt.cm.ScalarMappable(norm=mplt.colors.Normalize(vmin=0, vmax=max(E)), cmap='Reds')
    for t, e in zip(self.mesh.triangles, E):
      x, y = tuple(zip(*((n.x, n.y) for n in t.nodes)))
      ax.fill(x, y, color=scalar_map.to_rgba(e))
    
    return figax 

class Imposed_Fracture_Solution(Solution):
  def __init__(self, mesh, physical_data, fracture, time=1, tip_enrichement=False):
    self._tip_enrichement = tip_enrichement
    if tip_enrichement:
      self.mesh = msh.Broken_Mesh_Nonlinear_Tip(fracture, mesh)
    else:
      self.mesh = msh.Broken_Mesh_Linear_Tip(fracture, mesh)
    self.fracture = fracture
    self.field = el.Enriched_Finite_Elements(self.mesh)
    super().__init__(physical_data, self.field, time)

  @classmethod
  def from_classical_solution(cls, classical_solution, physical_data, fracture, time=1, tip_enrichement=False):
    """
    Using the classical solution, a lot of calculations might be avoided (i.e. majority of stiff_matrix coefficient might be keeped).
    """
    self = cls.__new__(cls)
    self._tip_enrichement = tip_enrichement
    self._time = time
    if tip_enrichement:
      self.mesh = msh.Broken_Mesh_Nonlinear_Tip(fracture, classical_solution.mesh)
    else:
      self.mesh = msh.Broken_Mesh_Linear_Tip(fracture, classical_solution.mesh)
    self.fracture = fracture
    self.field = el.Enriched_Finite_Elements.from_classical_field(classical_solution.field, self.mesh)
    self.physical_data = physical_data
    self.stiff_matrix = stiff.modify_stiff_matrix(classical_solution._sparse_stiff_data, self.field, physical_data.stiffness_tensor)
    self.u_bound = np.array(list(map(lambda p: physical_data.boundary_displacement(time, p), [e.base_node for e, e_ in zip(*[iter(self.field.boundary_elements)]*2)]))).flatten()
    self.boundary_matrix = stiff.compute_boundary_matrix(self.field, self.physical_data.stiffness_tensor)
    self.boundary_stiff = stiff.compute_boundary_stiff(self.field, self.physical_data.stiffness_tensor)
    self._solve()
    return self

  def plot_displacement(self, figax=None):
    interior_elements = self.field.interior_elements
    boundary_elements = self.field.boundary_elements
    
    if not figax:
      figax = plt.subplots(1)
    fig, ax = figax
    
    affine, mid_left, mid_right = [], [], []
    for i, element in enumerate(interior_elements + boundary_elements):
      if element.vector == "u_y":
        continue
      node = element.base_node
      u, v = self.u[i], self.u[i+1]
      if type(element) is el.Affine_Element:
        affine.append((node.x, node.y, u, v))
      elif type(element) is el.Mid_Element:
        if element.side < 0:
          mid_left.append((node.x, node.y, u, v))
        else:
          mid_right.append((node.x, node.y, u, v))
    
    #ax.figure(solution_id)
    X, Y, U, V = tuple(zip(*affine))
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy')#, scale=3)
    
    #X, Y, U, V = tuple(zip(*mid_left))
    #axs[1].quiver(X, Y, U, V, angles='xy', scale_units='xy')#, scale=3)
    #axs[1].set_xlim(axs[0].get_xlim())
    #axs[1].set_ylim(axs[0].get_ylim())
    #X, Y, U, V = tuple(zip(*mid_right))
    #axs[2].quiver(X, Y, U, V, angles='xy', scale_units='xy')#, scale=3)
    #axs[2].set_xlim(axs[0].get_xlim())
    #axs[2].set_ylim(axs[0].get_ylim())
    return figax
  
  def plot_energy(self, figax=None):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    
    E = []
    for t in self.mesh.triangles:
      elements = [e for n in t.nodes for e in self.field.elements_on_node[n]]
      coefficients = [self.u_int[self.field.interior_elements.index(e)] if e in self.field.interior_elements else self.u_bound[self.field.boundary_elements.index(e)] for e in elements]
      energy = 0
      for e1, c1, in zip(elements, coefficients):
        for e2, c2 in zip(elements, coefficients):
          energy += c1*c2*scalar_product.scalar_product_over_triangle(t, self.physical_data.stiffness_tensor, e1, e2)
      E.append(energy/t.area)
    
    scalar_map = mplt.cm.ScalarMappable(norm=mplt.colors.Normalize(vmin=0, vmax=max(E)), cmap='Reds')
    for t, e in zip(self.mesh.triangles, E):
      x, y = tuple(zip(*((n.x, n.y) for n in t.nodes)))
      ax.fill(x, y, color=scalar_map.to_rgba(e))
    
    return figax 


############
# EVALUATORS
############
class Evaluator:
  def __init__(self, discretization_data, physical_data, log_queue, classical_solution=None):
    self.discretization_data = discretization_data
    self.physical_data = physical_data
    self.log_queue = log_queue
    self.classical_solution = classical_solution

  def __call__(self, fracture_iterator, time):
    list_computations = []
    fractures = []
    while True:
      try:
        fracture = next(fracture_iterator)
      except StopIteration:
        break
      except msh.MultipleFracture:
        self.log_queue.put(('WARNING', 'Multiple Fracture error'))
      except Exception as e:
        self.log_queue.put(('ERROR', 'Unexpected error {} while computing next fracture'.format(e)))
      else:
        fractures.append(fracture)
    
    for fracture in fractures:
      solution_ = self._solve_fixed_fracture(fracture, time)
      if solution_.energy < np.inf:
        list_computations.append(Lazy_Solution(solution_))
    list_computations.sort(key=lambda x: x.energy)
    return list_computations

  def _solve_fixed_fracture(self, fracture, time):
    try:
      if self.classical_solution:
        solution = Imposed_Fracture_Solution.from_classical_solution(self.classical_solution, self.physical_data, fracture, time, self.discretization_data.tip_enrichement)
      else:
        solution = Imposed_Fracture_Solution(self.discretization_data.mesh, self.physical_data, fracture, time, self.discretization_data.tip_enrichement)
    except msh.ShouldBeAdmissibleFracture:
      self.log_queue.put(('WARNING', 'Should be admissible fracture {}'.format(fracture)))
      solution = Infinite_Energy
    except msh.NotAdmissibleFracture:
      self.log_queue.put(('DEBUG', 'Not admissible fracture {}'.format(fracture)))
      solution = Infinite_Energy
    except NumericalError:
      self.log_queue.put(('DEBUG', 'Numerical error for fracture {}'.format(fracture)))
      solution = Infinite_Energy
    except UmfpackWarning:
      self.log_queue.put(('DEBUG', 'Ill-conditioned matrix for fracture {}'.format(fracture)))
      solution = Infinite_Energy
    except Exception as e:
      self.log_queue.put(('ERROR', 'Unexpected error {} with fracture {}'.format(e, fracture)))
      solution = Infinite_Energy
    else:
      self.log_queue.put(('FULLDEBUG', 'Fracture {} OK !'.format(fracture)))
    
    return solution


class Evaluator_Multiprocess:
  def __init__(self, discretization_data, physical_data, log_queue, nbr_processes, classical_solution=None):
    self.discretization_data = discretization_data
    self.physical_data = physical_data
    self.log_queue = log_queue
    self.fracture_queue = Queue()
    self.result_queue = Queue()
    self.nbr_processes = nbr_processes
    self.classical_solution = classical_solution
  
  def __call__(self, fracture_iterator, time):
    list_computations = []
    fracture_iterator_process = Fracture_Iterator_Process(fracture_iterator, self.fracture_queue, self.nbr_processes, self.log_queue)
    fracture_iterator_process.start()
    fracture_solver_processes = []

    for i in range(self.nbr_processes):
      fsp = Fracture_Solver_Process(self.classical_solution, self.discretization_data, self.physical_data, time, self.fracture_queue, self.result_queue, self.log_queue, process_id=i)
      fsp.start()
      fracture_solver_processes.append(fsp)
    
    i = 0
    while i < self.nbr_processes:
      solution_ = self.result_queue.get()
      if solution_ is None:
        i += 1
      elif solution_.energy < np.inf:
        list_computations.append(solution_)
    
    for fsp in fracture_solver_processes:
      fsp.join(0)
    fracture_iterator_process.join(0)

    list_computations.sort(key=lambda x: x.energy)
    return list_computations


class Fracture_Iterator_Process(Process):
  def __init__(self, fracture_iterator, fracture_queue, nbr_processes, log_queue):
    Process.__init__(self)
    self.fracture_iterator = fracture_iterator
    self.fracture_queue = fracture_queue
    self.nbr_processes = nbr_processes
    self.log_queue = log_queue
  
  def run(self):
    while True:
      try:
        fracture = next(self.fracture_iterator)
      except StopIteration:
        break
      except msh.MultipleFracture:
        self.log_queue.put(('WARNING', 'Multiple Fracture error'))
      except Exception as e:
        self.log_queue.put(('ERROR', 'Unexpected error {} while computing next fracture'.format(e)))
      else:
        self.fracture_queue.put(msh.Lazy_Fracture(fracture))
    for i in range(self.nbr_processes):
      self.fracture_queue.put(None)


class Fracture_Solver_Process(Process):
  def __init__(self, classical_solution, discretization_data, physical_data, time, fracture_queue, result_queue, log_queue, process_id):
    Process.__init__(self)
    self.classical_solution = classical_solution
    self.discretization_data = discretization_data
    self.time = time
    self.fracture_queue = fracture_queue
    self.result_queue = result_queue
    self.physical_data = physical_data
    self.log_queue = log_queue
    self.process_id = process_id

  def run(self):
    while True:
      try:
        fracture = self.fracture_queue.get(1)
      except EmptyQueue:
        self.log_queue.put(('ERROR', 'No fracture to compute'))
        break
      if fracture is None:
        break
      
      try:
        fracture = fracture.to_full(self.discretization_data.mesh)
      except Exception as e:
        self.log_queue.put(('ERROR', 'Unexpected error with fracture {}'.format(e)))
        continue
        
      try:
        if self.classical_solution:
          solution = Imposed_Fracture_Solution.from_classical_solution(self.classical_solution, self.physical_data, fracture, self.time, self.discretization_data.tip_enrichement)
        else:
          solution = Imposed_Fracture_Solution(self.discretization_data.mesh, self.physical_data, fracture, self.time, self.discretization_data.tip_enrichement)
      
      except msh.ShouldBeAdmissibleFracture:
        self.log_queue.put(('WARNING', 'Should be admissible fracture {}'.format(fracture)))
        solution = Infinite_Energy
      except msh.NotAdmissibleFracture:
        self.log_queue.put(('DEBUG', 'Not admissible fracture {}'.format(fracture)))
        solution = Infinite_Energy
      except NumericalError:
        self.log_queue.put(('DEBUG', 'Numerical error for fracture {}'.format(fracture)))
        solution = Infinite_Energy
      except UmfpackWarning:
        self.log_queue.put(('DEBUG', 'Ill-conditioned matrix for fracture {}'.format(fracture)))
        solution = Infinite_Energy
      except Exception as e:
        self.log_queue.put(('ERROR', 'Unexpected error with fracture {}'.format(fracture)))
        solution = Infinite_Energy
      else:
        self.log_queue.put(('FULLDEBUG', 'Fracture {} OK !'.format(fracture)))
      
      if solution.energy < np.inf:
        self.result_queue.put(solution.strip_to_lazy_solution())
    
    self.result_queue.put(None)


########
# Solver
########
class Solver:
  def __init__(self, discretization_data, physical_data, log_queue, nbr_processes=None):
    self.discretization_data = discretization_data
    self.physical_data = physical_data
    self.log_queue = log_queue
    self.classical_solution = Classical_Solution(discretization_data.mesh, physical_data)
    if nbr_processes:
      self.evaluator = Evaluator_Multiprocess(discretization_data, physical_data, log_queue, nbr_processes, self.classical_solution)
    else:
      self.evaluator = Evaluator(discretization_data, physical_data, log_queue, self.classical_solution)
  
  def _get_fracture_iterator(self, old_solution):
    if type(old_solution.fracture) is not msh.No_Fracture:
      fracture = old_solution.fracture
      if type(fracture) is msh.Lazy_Fracture:
        fracture = fracture.to_full(self.discretization_data.mesh)
      if self.discretization_data.fracture_discretization.interior_fast_step:
        fracture_iterator = fracture_iterators.Admissible_Fractures_From_Fracture_Fast(self.discretization_data.fracture_discretization, self.discretization_data.mesh, fracture)
      else:
        fracture_iterator = fracture_iterators.Admissible_Fractures_From_Fracture(self.discretization_data.fracture_discretization, self.discretization_data.mesh, fracture)
    else:
      fracture_iterator = iter([])
      if self.physical_data.initial_fracture:
        fracture = msh.Fracture.from_str(self.physical_data.initial_fracture, self.discretization_data.mesh)
        if self.discretization_data.fracture_discretization.interior_fast_step:
          fracture_iterator = fracture_iterators.Admissible_Fractures_From_Fracture_Fast(self.discretization_data.fracture_discretization, self.discretization_data.mesh, fracture)
        else:
          fracture_iterator = fracture_iterators.Admissible_Fractures_From_Fracture(self.discretization_data.fracture_discretization, self.discretization_data.mesh, fracture)
      else:
        if self.discretization_data.fracture_discretization.boundary_step:
          fracture_iterator = itertools.chain(fracture_iterator, fracture_iterators.Admissible_Fractures_From_Boundary(self.discretization_data.fracture_discretization, self.discretization_data.mesh))
        elif self.discretization_data.fracture_discretization.boundary_point:
          fracture_iterator = itertools.chain(fracture_iterator, fracture_iterators.Admissible_Fractures_From_Fixed_Boundary_Point(self.discretization_data.fracture_discretization, self.discretization_data.mesh))
        if self.discretization_data.fracture_discretization.interior_step:
          fracture_iterator = itertools.chain(fracture_iterator, fracture_iterators.Admissible_Fractures_From_Interior(self.discretization_data.fracture_discretization, self.discretization_data.mesh))

    return fracture_iterator
  

class Solver_With_Time_Discretization(Solver):
  """
  Exploits the linear structure, and calls the Evaluator class when needed
  """
  def __init__(self, discretization_data, physical_data, log_queue, nbr_processes=None):
    super().__init__(discretization_data, physical_data, log_queue, nbr_processes)
    self.time_discretization = discretization_data.time_discretization
    self.list_times = []
    self.list_computations = []
    if isinstance(self.physical_data.boundary_displacement, problem_data.Linear_Displacement):
      self._fixed_time_solver = self._fixed_time_solver_linear
    else:
      self._fixed_time_solver = self._fixed_time_solver_nonlinear

  def solve(self):
    solution = self.classical_solution
    for time in self.time_discretization:
      solution = self._fixed_time_solver(solution, time)
      self.log_queue.put(('STATUS', "Time {:4f} {}".format(time, solution.fracture)))
      if solution.fracture.is_traversant:
        break 
    
    if type(solution) is Lazy_Solution:
      solution = solution.to_full(self.discretization_data.mesh, self.physical_data)
    return solution
  
  def _fixed_time_solver_nonlinear(self, old_solution, time):
    fracture_iterator = self._get_fracture_iterator(old_solution)
    solution = self.evaluator(fracture_iterator, time)[0]
    old_solution.change_time(time)
    if old_solution.energy < solution.energy:
      return old_solution
    else:
      return solution

  def _fixed_time_solver_linear(self, old_solution, time):
    self.list_times.append(time)
    if len(self.list_times) < 2 or not self.list_computations:
      fracture_iterator = self._get_fracture_iterator(old_solution)
      self.list_computations = self.evaluator(fracture_iterator, time)
      solution = self.list_computations[0]
      if solution.energy > old_solution.energy:
        return old_solution
      else:
        self.list_computations = []
        return solution
    else:
      old_solution.change_time(time)
      for solution_ in self.list_computations:
        solution_.change_time(time)
      self.list_computations.sort(key=lambda x: x.energy)
      solution = self.list_computations[0]
      if solution.energy > old_solution.energy:
        solution = old_solution
      else:
        self.list_computations = []
      return solution


class Smart_Time_Solver(Solver):
  def __init__(self, discretization_data, physical_data, log_queue, nbr_processes=None):
    super().__init__(discretization_data, physical_data, log_queue, nbr_processes)
    assert isinstance(physical_data.boundary_displacement, problem_data.Linear_Displacement)

  def solve(self):
    time = 0
    solution = self.classical_solution
    while time < 1 and not solution.fracture.is_traversant:
      fracture_iterator = self._get_fracture_iterator(solution)
      time_, solution_ = self._next_fracture(time, solution, fracture_iterator)
      if time_ == 1:
        break
      else:
        time, solution = time_, solution_
        self.log_queue.put(('STATUS', "Time {} {}".format(time, solution.fracture)))
    
    if type(solution) is Lazy_Solution:
      solution = solution.to_full(self.discretization_data.mesh, self.physical_data)
    return solution

  def _next_fracture(self, old_time, old_solution, fracture_iterator):
    if old_time == 0:
      old_time = 1
      list_computations = self.evaluator(fracture_iterator, time=old_time)
    else:
      list_computations = self.evaluator(fracture_iterator, time=old_time)
      # Check if infinite-speed propagation (no physical sence, but happens with that algorithm)
      solution = list_computations[0]
      if solution.energy < old_solution.energy:
        return old_time, solution
      
    #FIXME : should not happend mathematically, but happens numerically
    list_computations = [x for x in list_computations if x.elastic_energy < old_solution.elastic_energy]
    
    sort_func = lambda x: sqrt((x.fracture_energy - old_solution.fracture_energy)/(old_solution.elastic_energy - x.elastic_energy))
    list_computations.sort(key=sort_func)
    
    if list_computations:
      solution = list_computations[0]
      time = old_time*sort_func(solution)
      if time > 1: 
        return 1, old_solution
      else:
        solution.change_time(time)
        return time, solution
    else:
      return 1, old_solution


def smart_time_solver(discretization_data, physical_data, log_queue, nbr_processes=None):
  solver = Smart_Time_Solver(discretization_data, physical_data, log_queue, nbr_processes)
  return solver.solve()


def solver_with_time_discretization(discretization_data, physical_data, log_queue, nbr_processes=None):
  solver = Solver_With_Time_Discretization(discretization_data, physical_data, log_queue, nbr_processes)
  return solver.solve()

