import sys, os
import numpy as np
from math import pi, exp, sqrt

from .geometry import Point, Circle, rotation, dist
sys.path.append(os.path.abspath('..'))
import plotrc as plot_options


#####################
# TIME DISCRETIZATION
#####################
class Regular_Time_Discretization:
  def __init__(self, time_step, time_min=None):
    self._time_step = time_step
    self._time = time_min if time_min else time_step
    self._raise = False

  def __iter__(self):
    return self

  def __next__(self):
    time = self._time
    self._time += self._time_step
   
    if self._raise:
      raise StopIteration
    
    if time > 1:
      time = 1
      self._raise = True
    return time


#########################
# FRACTURE DISCRETIZATION
#########################
class Fracture_Discretization:
  def __init__(self, angular_step, lengh_step, boundary_step=None, boundary_point=None, interior_step=None, interior_fast_init=True, interior_fast_step=True, interior_start_angle=None):
    self.angular_step = angular_step
    self.lengh_step = lengh_step
    self.min_lengh = lengh_step #FIXME
    self.max_lengh = np.inf #FIXME
    
    self.interior_step = interior_step
    self.interior_fast_step = interior_fast_step
    self.interior_fast_init = interior_fast_init
    if interior_start_angle:
      self.interior_start_angle = interior_start_angle
    else:
      self.interior_start_angle = 1.57079632679 # pi/2
    self.boundary_step = boundary_step
    if boundary_point:
      self.boundary_point = Point(*boundary_point)
      assert not boundary_step
    else:
      self.boundary_point = None


#######################
# BOUNDARY DISPLACEMENT
#######################
class Boundary_Displacement:
  """
  Abstract class for boundary displacement.
  Personalized boundary displacement should inherit from it, and implement __call__ function.
  """
  def __call__(self, p):
    raise NotImplementedError


class Linear_Displacement(Boundary_Displacement):
  def __init__(self, traction_coefficient=1):
    self._traction_coefficient = traction_coefficient
  
  def __call__(self, time, p):
    return self._traction_coefficient*time*self._func(p)


class Constant_Displacement_On_Y(Linear_Displacement):
  r"""
  | | -> |     |
  """
  def __init__(self, traction_coefficient=1, abscissa_mid=0.9):
    super().__init__(traction_coefficient) 
    self._abscissa_mid = abscissa_mid

  def _func(self, p):
    if p.x > self._abscissa_mid:
      return np.array([1, 0])
    return np.array([0, 0])


class Linear_Displacement_On_Y(Linear_Displacement):
  r"""
  | | -> / \
  """
  def __init__(self, traction_coefficient=1, abscissa_mid=0.5, y_min=0, c_min=1, y_max=100, c_max=0):
    super().__init__(traction_coefficient)
    self._abscissa_mid = abscissa_mid
    self.c_min, self.c_max = c_min, c_max 
    self.y_min, self.y_max = y_min, y_max

  def _func(self, p):
    c_y = self.c_min + (self.c_max - self.c_min)/(self.y_max - self.y_min)*p.y
    if p.x > self._abscissa_mid:
      return np.array([c_y, 0])
    else:
      return np.array([-c_y, 0])


class Picewise_Linear_Displacement_On_Y(Linear_Displacement):
  r"""
  | | -> ||
  | | -> / \
  """
  def __init__(self, traction_coefficient=1, abscissa_mid=0.5, y_min=0, y_max=100):
    super().__init__(traction_coefficient)
    self._abscissa_mid = abscissa_mid
    self._y_min = y_min
    self._y_max = y_max
    self._amplitude = y_max - y_min

  def _func(self, p):
    if p.y > self._y_max:
      return np.array([0, 0])
    if p.x > self._abscissa_mid:
      return np.array([(self._y_max - p.y)/self._amplitude, 0])
    else:
      return np.array([-(self._y_max - p.y)/self._amplitude, 0])


class Rotation_Displacement_On_Y(Boundary_Displacement):
  r"""
  #| | -> _ _ (angle = pi/2) 
  #| | -> / \ (angle = pi/4)
  This displacement is not linear.
  """
  def __init__(self, angle=1, abscissa_mid=0.5, point_left=(0, 100), point_right=(100, 100)):
    self._angle = angle
    self._abscissa_mid = abscissa_mid
    self._point_left = Point(*point_left)
    self._point_right = Point(*point_right)

  def __call__(self, time, p):
    if p.x>self._abscissa_mid:
      return (rotation(self._point_right, p, self._angle*time) - p).array
    else:
      return (rotation(self._point_left, p, -self._angle*time) - p).array


##############
# STIFF TENSOR
##############
class Identity_Tensor:
  def __init__(self):
    pass
  
  def tproduct(self, matrix):
    return matrix


class Lame_Tensor:
  def __init__(self, lambda_, mu):
    self._lambda = lambda_
    self._mu = mu
  
  def tproduct(self, matrix):
    return 2*self._mu*matrix + self._lambda*matrix.trace()*np.array(((1, 0),(0, 1)))
  
  @classmethod
  def _init_with_Young_Poisson(cls, E, nu):
    return cls(E*nu/((1+nu)*(1-2*nu)), 0.5*E/(1+nu))

# Classical ice tensor
lame_tensor_ice = Lame_Tensor._init_with_Young_Poisson(8.95, 0.295)


#################
# TOUGHNESS FIELD
#################
class Constant_Toughness:
  def __init__(self, k):
    self.k = k
    
  def __call__(self, p):
    return self.k
  
  def plot(self, figax):
    pass
    

class Japan_Toughness:
  def __init__(self, k1, k2, center, radius):
    self.k1= k1
    self.k2 = k2
    self.circle = Circle(Point(*center), radius)
  
  def __call__(self, p):
    if self.circle.has_point(p):
      return self.k2
    else:
      return self.k1
  
  def plot(self, figax=None):
    self.circle.plot(figax, **plot_options.circle_inclusion)
    return figax


class Smooth_Japan_Toughness(Japan_Toughness):
  def __init__(self, k1, k2, center, radius):
    super().__init__(k1, k2, center, radius)
    self.sigma = 0.25
    
  def __call__(self, p):
    d = dist(p, self.circle.center)
    if d < self.circle.radius:
      return self.k1 + (self.k2 - self.k1)*exp(-(self.circle.radius - d)**2/2*self.sigma**2)#/(self.sigma*sqrt(2*pi))
    else:
      return self.k1


##############
# PROBLEM DATA
##############
class Discretization_Data:
  def __init__(self, mesh, time_discretization, fracture_discretization, tip_enrichement=False):
    self.mesh = mesh
    self.time_discretization = time_discretization
    self.fracture_discretization = fracture_discretization
    self.tip_enrichement = tip_enrichement

    if fracture_discretization.boundary_point:
      assert mesh.has_point_on_boundary(self.fracture_discretization.boundary_point)


class Physical_Data:
  def __init__(self, stiffness_tensor, toughness_field, boundary_displacement, initial_fracture=None):
    self.stiffness_tensor = stiffness_tensor
    self.toughness_field = toughness_field
    self.boundary_displacement = boundary_displacement
    self.initial_fracture = initial_fracture
