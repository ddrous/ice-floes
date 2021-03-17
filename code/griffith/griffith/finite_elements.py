import matplotlib.pyplot as plt
from math import sqrt, sin, cos, atan2, pi
import numpy as np
from enum import Enum

from . import geometry as geo
from . import mesh as msh


class Element:
  """
  Vector is worth either 'u_x' or 'u_y', and is used mainly at construction and for plotting.
  """
  Type = Enum('Type', 'INTERIOR BOUNDARY')
  def __init__ (self, base_node, vector):
    self.base_node = base_node
    self.vector = vector
    
  @property
  def type(self):
    assert type(self._type) is self.Type
    return self._type

  @type.setter
  def type(self, new_type):
    self._type = new_type

  @property
  def index(self):
    assert self._index != None
    return self._index

  @index.setter
  def index(self, new_index):
    self._index = new_index
  
  def __repr__ (self):
    return "Element of type {}, on node {}, and with vector {}".format(type(self), self.base_node, self.vector)


class Affine_Element(Element):
  """
  The attribute gradients is a dictionnary indexed by the triangle.
  On the triangle t (wich we assmue is a neighboor of self.node), we compute the gradient of the element active our node, and store {t : value}
  """
  def __init__ (self, base_node, vector):
    super().__init__(base_node, vector)
    self._dict_gradient = {} # faster
    self._dict_sym_gradient = {}
    self._dict_function = {}
    self._index = None
    self.type = None

    for t in self.base_node.of_triangles:
      self._dict_gradient[t.id_number] = self._gradient_on_triangle(t)
      self._dict_sym_gradient[t.id_number] = 0.5*(self._dict_gradient[t.id_number] + np.transpose(self._dict_gradient[t.id_number]))
  
  def gradient(self, point, triangle=None, *args, **kwargs):
    if triangle:
      return self._dict_gradient[triangle.id_number]
    for t in self.base_node.of_triangles:
      if t.has_point(point):
        return self._dict_gradient[t.id_number]

    return np.array(([0, 0], [0, 0]))
  
  def sym_gradient(self, point, triangle=None, *args, **kwargs):
    if triangle:
      return self._dict_sym_gradient[triangle.id_number]
    else:
      for t in self.base_node.of_triangles:
        if t.has_point(point):
          return self._dict_sym_gradient[t.id_number]
    
  def _gradient_on_triangle (self, triangle):
    """
    Compute the gradient over triangle
    """
    if triangle not in self.base_node.of_triangles:
      return np.array(([0, 0], [0, 0]))
    
    non_active_nodes = [node for node in triangle.nodes if node is not self.base_node]
    # D is the oriented area of the triangle
    D = np.linalg.det(np.array([[self.base_node.x, self.base_node.y, 1],[non_active_nodes[0].x, non_active_nodes[0].y, 1],[non_active_nodes[1].x, non_active_nodes[1].y, 1]]))
    gradient = np.array([non_active_nodes[0].y-non_active_nodes[1].y, non_active_nodes[1].x-non_active_nodes[0].x]) / D
    if self.vector == "u_x":
      return np.array((gradient, [0 , 0]))
    elif self.vector == "u_y":
      return np.array(([0 , 0], gradient))
  
  def function(self, point, triangle=None):
    try:
      coeff = self._dict_function[(point.x, point.y)]
    except KeyError:
      if triangle:
        coeff = self._function_over_triangle(point, triangle)
      elif self.base_node.is_eq(point):
        coeff = 1
      else:
        for t in self.base_node.of_triangles:
          if t.has_point(point):
            coeff = self._function_over_triangle(point, t)
    
    if self.vector == "u_x":
      return np.array([coeff, 0])
    elif self.vector == "u_y":
      return np.array([0, coeff])
  
  def _function_over_triangle(self, point, t):
    n1, n2 = set(t.nodes) - set((self.base_node,))
    n3 = self.base_node
    normal_1 = [point.y - n3.y, point.x - n3.x] # normal vector of the line (n3, point)
    normal_2 = [n2.y - n1.y, n2.x - n1.x] # normal vector of the line (n1, n2)

    N = [n3.x*normal_1[0] + n3.y*normal_1[1], n2.x*normal_2[0] + n2.y*normal_2[1]]
    # inversion of the matrix [normal_1, normal_2]
    # M = 1/(normal_1[0]*normal_2[1] - normal_2[0]*normal_1[1]) * np.array([[normal_2[1], -normal_1[1]], [-normal_2[0], normal_1[0]]]) 
    # inter = M.dot(N)
    det = 1/(normal_1[0]*normal_2[1] - normal_2[0]*normal_1[1])
    inter = [det*(normal_2[1]*(n3.x*normal_1[0] + n3.y*normal_1[1]) + -normal_1[1]*(n2.x*normal_2[0] + n2.y*normal_2[1])),
             det*(-normal_2[0]*(n3.x*normal_1[0] + n3.y*normal_1[1]) + normal_1[0]*(n2.x*normal_2[0] + n2.y*normal_2[1]))]

    coeff = geo.dist(self.base_node, point)/sqrt((inter[0] - n3.x)**2 + (inter[1] - n3.y)**2)
    
    self._dict_function[(point.x, point.y)] = coeff
    return coeff


class Mid_Element(Affine_Element):
  def __init__ (self, element, side, local_refinement):
    self.base_node = element.base_node
    self.vector = element.vector
    self._dict_function = element._dict_function
    self._dict_gradient = element._dict_gradient
    self._dict_sym_gradient = element._dict_sym_gradient
    self.side = side
    self._local_refinement = local_refinement
    self._index = None
    self.type = element.type

  def local_refinement(self, t):
    return self._local_refinement[t][self.side]

  def __repr__(self):
    return super().__repr__() + " and on side {}".format(self.side)


class Finite_Elements:
  """
  A FiniteElements object contains four important attributes :
  - a list of p1 interior elements
  - a list of p1 boundary elements
  - a dictonary for the correspondance node <-> element
  - a list of the associated boundary values
  """
  def __init__(self, mesh):
    self.mesh = mesh
    self.interior_elements = []
    self.boundary_elements = []
    self.elements_on_node = {}

    for node in mesh.nodes:
      element_ux, element_uy = Affine_Element (node, "u_x"), Affine_Element (node, "u_y")
      if mesh.is_node_dirichlet(node):
        element_ux.type, element_uy.type = Element.Type.BOUNDARY, Element.Type.BOUNDARY
        element_ux.index, element_uy.index = len(self.boundary_elements), len(self.boundary_elements) + 1
        self.boundary_elements.extend([element_ux, element_uy])
      else:
        element_ux.type, element_uy.type = Element.Type.INTERIOR, Element.Type.INTERIOR
        element_ux.index, element_uy.index = len(self.interior_elements), len(self.interior_elements) + 1
        self.interior_elements.extend([element_ux, element_uy])
      self.elements_on_node[node] = [element_ux, element_uy]


class Enriched_Finite_Elements(Finite_Elements):
  """
  Adds fractured_elements to the element list.
  The removed_element_list is for the stiff matrix computation.
  """
  def __init__(self, mesh):
    super().__init__(mesh)
    self.removed_interior_indexes = []
    self.new_interior_elements = []
    self.modified_interior_elements = []
    self.old_to_new_indexes = []
    self._init_fracture_elements()

  @classmethod
  def from_classical_field(cls, field, broken_mesh):
    """
    We need to modify the field first.
    """
    self = cls.__new__(cls) 
    self.interior_elements = field.interior_elements.copy()
    self.boundary_elements = field.boundary_elements.copy()
    self.elements_on_node = {key: value.copy() for key, value in field.elements_on_node.items()}
    self.mesh = broken_mesh
    self.removed_interior_indexes = []
    self.new_interior_elements = []
    self.modified_interior_elements = []
    self.old_to_new_indexes = []
    
    # Add Element for tip affine enrichement
    if type(self.mesh) is msh.Broken_Mesh_Linear_Tip and self.mesh._tip_enrichement:
      for tip_node in self.mesh._extra_tip_nodes:
        element_ux, element_uy = Affine_Element (tip_node, "u_x"), Affine_Element (tip_node, "u_y")
        self.interior_elements.extend([element_ux, element_uy])
        self.elements_on_node[tip_node] = [element_ux, element_uy] 
        element_ux.type, element_uy.type = Element.Type.INTERIOR, Element.Type.INTERIOR
      
      for old, new in self.mesh._old_and_new_tip_nodes:
        element_ux, element_uy = Affine_Element (new, "u_x"), Affine_Element (new, "u_y")
        self.elements_on_node[new] = [element_ux, element_uy]
        old_element_ux, old_element_uy = self.elements_on_node[old]
        element_ux.index, element_uy.index = old_element_ux.index, old_element_uy.index
        if old_element_ux in self.interior_elements:
          self.interior_elements[self.interior_elements.index(old_element_ux)] = element_ux
          self.interior_elements[self.interior_elements.index(old_element_uy)] = element_uy
          element_ux.type, element_uy.type = Element.Type.INTERIOR, Element.Type.INTERIOR
          if new not in self.mesh.mid_nodes:
            self.modified_interior_elements.extend([element_ux, element_uy])
        else:
          self.boundary_elements[self.boundary_elements.index(old_element_ux)] = element_ux
          self.boundary_elements[self.boundary_elements.index(old_element_uy)] = element_uy
          element_ux.type, element_uy.type = Element.Type.BOUNDARY, Element.Type.BOUNDARY
    
    self._init_fracture_elements()
    return self
  
  def _init_fracture_elements(self):
    """
    We add the fractured elements at the end on the field.elements list.
    """
    # Store indexes before changes
    for i, element in enumerate(self.interior_elements):
      if element.base_node in self.mesh.mid_nodes:
        if type(self.mesh) is msh.Broken_Mesh_Linear_Tip and element.base_node in self.mesh._extra_tip_nodes: # the new_tip_nodes are at the end of the list
            continue
        self.removed_interior_indexes.append(i)
    
    self.removed_interior_indexes.sort()
    self.modified_interior_indexes = [e.index for e in self.modified_interior_elements]
    
    # Mid elements
    mid_elements_on_this_node = []
    l = list(self.mesh.mid_nodes)
    l.sort(key=lambda x:x.id_number)
    for node in l:
      # Initialize the Mid_Element
      for element in self.elements_on_node[node]:
        left = Mid_Element(element, -1.0, self.mesh.local_refinement)
        right = Mid_Element(element, 1.0, self.mesh.local_refinement)
        mid_elements_on_this_node.extend([left, right])
        try:
          self.interior_elements.remove(element)
        except ValueError:
          boundary, interior = left, right
          points_interior = set([point for t in node.of_triangles for tt in interior.local_refinement(t) for point in tt.points])
          points_interior = points_interior.difference(set([point for t in node.of_triangles for tt in boundary.local_refinement(t) for point in tt.points]))
          for point in points_interior:
            if point in self.mesh.nodes and self.mesh.is_node_dirichlet(point):
              boundary, interior = interior, boundary
              break
          self.interior_elements.append(interior)
          self.new_interior_elements.append(interior)
          self.boundary_elements[self.boundary_elements.index(element)] = boundary
          boundary.type, interior.type = Element.Type.BOUNDARY, Element.Type.INTERIOR
        else:
          left.type, right.type = Element.Type.INTERIOR, Element.Type.INTERIOR
          self.interior_elements.extend([left, right])
          self.new_interior_elements.extend([left, right])
      self.elements_on_node[node] = mid_elements_on_this_node
      mid_elements_on_this_node = []

    j = 0
    for i, element_i in enumerate(self.interior_elements):
      element_i.index = i
      if i not in self.removed_interior_indexes:
        self.old_to_new_indexes.append(j)
        j += 1
      else:
        self.old_to_new_indexes.append(None)
    
    for i, element_i in enumerate(self.boundary_elements):
      element_i.index = i

