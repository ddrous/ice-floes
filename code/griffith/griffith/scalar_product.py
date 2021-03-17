import numpy as np

from . import geometry as geo
from . import finite_elements as el


def scalar_product(stiffness_tensor, e1, e2):
  """
  Scalar product of two finite element function
  """
  triangles = e1.base_node.of_triangles & e2.base_node.of_triangles
  result = 0
  for t in triangles:
    result += scalar_product_over_triangle(t, stiffness_tensor, e1, e2)

  return result

def scalar_product_over_triangle(t, stiffness_tensor, e1, e2):
  result = 0
  
  # Both are affine elements
  if type(e1) is el.Affine_Element and type(e2) is el.Affine_Element:
      result = np.trace(np.dot(stiffness_tensor.tproduct(e1.sym_gradient(point=None, triangle=t)), e2.sym_gradient(point=None, triangle=t)))*t.area
  
  # One of e1, e2 is Mid_Element
  if type(e2) is el.Mid_Element:
    e1, e2 = e2, e1
  
  if type(e1) is el.Mid_Element:
    # Both are enriched elements
    if type(e2) is el.Mid_Element:
      for tt in e1.local_refinement(t).intersection(e2.local_refinement(t)):
        result += tt.area*np.trace(np.dot(stiffness_tensor.tproduct(e1.sym_gradient(point=None, triangle=t)), e2.sym_gradient(point=None, triangle=t)))

    # Only e1 is enriched element
    elif type(e2) is el.Affine_Element:
      for tt in e1.local_refinement(t):
        result += tt.area*np.trace(np.dot(stiffness_tensor.tproduct(e1.sym_gradient(point=None, triangle=t)), e2.sym_gradient(point=None, triangle=t)))
  
  return result 
  
