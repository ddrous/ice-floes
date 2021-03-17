import itertools
import math
from scipy.integrate import dblquad
import numpy as np

from . import geometry as geo

# Points were found on http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
# For the triangle (0,0), (0, 1), (1, 0)
gaussian_points_triangle = [(0.659027622374092, 0.231933368553031),
          (0.659027622374092, 0.109039009072877),
          (0.231933368553031, 0.659027622374092),
          (0.231933368553031, 0.109039009072877),
          (0.109039009072877, 0.659027622374092),
          (0.109039009072877, 0.231933368553031)]

gaussian_weights_triangle = [0.16666666666666666667]*6

# Points and weights taken from https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules/quadrature_rules.html
# and adapted for the square [0, 1]^2 ...
gaussian_points_square = [(0.11270166537925824, 0.11270166537925824),
                          (0.11270166537925824, 0.5),
                          (0.11270166537925824, 0.8872983346207417),
                          (0.5, 0.11270166537925824),
                          (0.5, 0.5),
                          (0.5, 0.8872983346207417),
                          (0.8872983346207417, 0.11270166537925824),
                          (0.8872983346207417, 0.5),
                          (0.8872983346207417, 0.8872983346207417)]

gaussian_weights_square = [0.0771604938271605,
                           0.1234567901234568,
                           0.0771604938271605,
                           0.1234567901234568,
                           0.19753086419753088,
                           0.1234567901234568,
                           0.0771604938271605,
                           0.1234567901234568,
                           0.0771604938271605]


def scalar_product_over_singular_triangle(elements, stiffness_tensor, triangle, sub_triangle, singular_point, duffin_coefficient):
  e1, e2 = elements
  p1 = singular_point
  p2, p3 = sub_triangle.other_points(p1)
  phi = lambda u, v: (p1.x + (p2.x - p1.x)*u + (p3.x - p2.x)*v, p1.y + (p2.y - p1.y)*u + (p3.y - p2.y)*v)
  duffin = lambda u, v: (u**duffin_coefficient, u**duffin_coefficient*v)
  jacobian_phi = math.fabs((p2.x - p1.x)*(p3.y - p1.y) - (p3.x - p1.x)*(p2.y - p1.y))
  jacobian_duffin = lambda u, v: duffin_coefficient*u**(2*duffin_coefficient-1)
  
  result = 0 
  for i, integration_point in enumerate(gaussian_points_square):
    maped_point = geo.Point(*phi(*duffin(*integration_point)))
    sym_gradient_1 = stiffness_tensor.tproduct(e1.sym_gradient(maped_point, triangle=triangle, integration_point=integration_point))
    sym_gradient_2 = e2.sym_gradient(maped_point, triangle=triangle, integration_point=integration_point)
    result += gaussian_weights_square[i]*jacobian_phi*jacobian_duffin(maped_point.x, maped_point.y)*np.trace(np.dot(sym_gradient_1, sym_gradient_2))
  return result


def scalar_product_over_regular_triangle(elements, stiffness_tensor, triangle, sub_triangle):
  e1, e2 = elements
  p1, p2, p3 = sub_triangle.points
  phi = lambda u, v: (p1.x + (p2.x - p1.x)*u + (p3.x - p1.x)*v, p1.y + (p2.y - p1.y)*u + (p3.y - p1.y)*v)
  detjac = math.fabs((p2.x - p1.x)*(p3.y - p1.y) - (p3.x - p1.x)*(p2.y - p1.y))
  
  result = 0
  for i, integration_point in enumerate(gaussian_points_triangle):
    maped_point = geo.Point(*phi(*integration_point))
    sym_gradient_1 = stiffness_tensor.tproduct(e1.sym_gradient(maped_point, triangle=triangle, integration_point=integration_point))
    sym_gradient_2 = e2.sym_gradient(maped_point, triangle=triangle, integration_point=integration_point)
    result += gaussian_weights_triangle[i]*detjac*np.trace(np.dot(sym_gradient_1, sym_gradient_2))
    
  # area of unit triangle
  return 0.5*result
