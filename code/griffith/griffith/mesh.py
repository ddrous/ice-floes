import sys, os
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum
import numpy as np
import re

from . import geometry as geo
sys.path.append(os.path.abspath('..'))
import plotrc as plot_options


class NotAdmissibleFracture(Exception):
  pass 


class ShouldBeAdmissibleFracture(Exception):
  pass


class MultipleFracture(Exception):
  pass


class Intersection_Data:
  """
  Base class for intersection types in Broken Mesh.
  """
  pass


class T1(Intersection_Data):
  def __init__(self, edge, segment, point):
    self.edge = edge
    self.segment = segment
    self.point = point


class T2(Intersection_Data):
  def __init__(self, edges, segment, point):
    self.edges = edges
    self.segment = segment
    self.point = point


class T3(Intersection_Data):
  def __init__(self, edge, segments, point):
    self.edge = edge
    self.segments = segments
    self.point = point


class T4(Intersection_Data):
  def __init__(self, edges, segments, point):
    self.edges = edges
    self.segments = segments
    self.point = point


class Node (geo.Point):
  """
  A node of the mesh. The positions, and the id are read in the mesh file.
  Here is a list of the principal attributes:
    - of_triangles is the set of triangles which have self as node.
    - neighboors is a set of neighbooring nodes (without self in it)
    - of_edges is for boundary_nodes only #WARNING 
  """
  def __init__ (self, x, y, id_number, of_triangles=None, neighboors=None, of_edges=None, list_of_triangles=None, list_of_nodes=None):
    geo.Point.__init__(self, x, y)
    self.id_number = id_number
    self._of_triangles = of_triangles if of_triangles else set()
    self._neighboors = neighboors if neighboors else set()
    self.of_edges = of_edges if of_edges else set()
    self._list_of_nodes = list_of_nodes
    self._list_of_triangles = list_of_triangles

  def _add_triangle(self, t_id):
    self._of_triangles.add(t_id)
  
  @property
  def of_triangles(self):
    return set((self._list_of_triangles[n] for n in self._of_triangles))

  def _add_neighboors(self, neighboors_id):
    self._neighboors.update(set((n_id for n_id in neighboors_id)) - set((self.id_number,)))
  
  @property
  def neighboors(self):
    return set((self._list_of_nodes[n] for n in self._neighboors))
  
  def _add_edge(self, edge):
    self.of_edges.add(edge)
  
  def __repr__ (self):
    return "Node {} : (x, y) = {}".format(self.id_number, (self.x, self.y))


class Triangle(geo.Triangle):
  def __init__ (self, node_1, node_2, node_3, id_number):
    super().__init__(node_1, node_2, node_3)
    self.id_number = id_number
    self.nodes = self.points

  def __repr__(self):
    return "Triangle {} on points {}".format(self.id_number, self.points)
  

class Edge(geo.Segment):
  """
  An edge of the mesh.
  The attribute group refers to the pyhsical group read in gmsh's .msh file.
  """
  def __init__(self, point_1, point_2, group):
    super().__init__(point_1, point_2)
    self.group = group

  def __repr_(self):
    return "Edge : between {} and {}".format(self.point_1, self.point_2)
  
  def swap_orientation(self):
    return Edge(self.point_2, self.point_1, self.group)


class Fracture(geo.Polyline):
  @classmethod
  def from_str(cls, str_fracture, mesh):
    points = str_fracture.replace('(', '').replace(',', '').replace(')', '').split()
    points = [float(p) for p in points]
    points = [geo.Point(x, y) for x, y in zip(*[iter(points)]*2)]
    segments = [geo.Segment(p1, p2) for p1, p2 in zip(points, points[1:])]
    return cls(segments, mesh)
    
  def __init__(self, segments, mesh):
    self.mesh = mesh
    
    # Merge segments if on same line
    self.segments = [segments[0]]
    for s2 in segments[1:]:
      s1 = self.segments[-1]
      if geo.are_vectors_colinear(s1.direction_vector, s2.direction_vector):
        self.segments[-1] = geo.Segment(s1.point_1, s2.point_2)
      else:
        self.segments.append(s2)

    if len(self.segments) == 1:
      intersections = self._intersections_with_boundary(self.first_segment)
      if len(intersections) == 0:
        start_point, end_point = None, None
      elif len(intersections) == 1:
        p, = intersections
        if self.mesh.has_point(self.start_point):
          start_point, end_point = None, p
        else:
          assert self.mesh.has_point(self.end_point)
          start_point, end_point = p, None
      else:
        p1, p2 = intersections
        if geo.dist(self.first_segment.point_1, p1) < geo.dist(self.first_segment.point_1, p2):
          start_point, end_point = p1, p2
        else:
          start_point, end_point = p2, p1
    
    else:
      s1, s2 = self.first_segment, self.last_segment
      if len(self.segments) > 2:
        for s in self.segments[1:-1]:
          if len(self._intersections_with_boundary(s)) > 0:
            raise MultipleFracture
      
      i1 = self._intersections_with_boundary(self.first_segment)
      if len(i1) > 1:
        raise MultipleFracture
      elif len(i1) == 1:
        start_point, = i1
      else:
        start_point = None
      
      i2 = self._intersections_with_boundary(self.last_segment)
      if len(i2) > 1:
        raise MultipleFracture
      elif len(i2) == 1:
        end_point, = i2
      else:
        end_point = None
        
    if start_point and end_point:
      self.is_traversant = True
    else:
      self.is_traversant = False
    
    if start_point:
      self.first_segment = geo.Segment(start_point, self.first_segment.point_2)
      self.first_tip_enrichement = False
    else:
      self.first_tip_enrichement = True
    if end_point:
      self.last_segment = geo.Segment(self.last_segment.point_1, end_point)
      self.last_tip_enrichement = False
    else:
      self.last_tip_enrichement = True
    
  def _intersections_with_boundary(self, segment):
    intersections = []
    for edge in self.mesh.boundary_mesh.edges:
      try:
        inter = geo.intersection(edge, segment)
      except geo.NoIntersection:
        continue
      except geo.OverlapError:
        raise MultipleFracture
      except geo.OnBoundary as e:
        inter = next(iter(e.intersection))
        if not any(inter.is_eq(p) for p in intersections):
          intersections.append(inter)
      else:
        intersections.append(inter)
    
    if len(intersections) > 2:
      raise MultipleFracture
    
    return intersections
 
  def plot(self, figax=None):
    if not figax:
      figax = plt.subplots()
      fig, ax = figax
    else:
      fig, ax = figax

    points = [segment.point_1 for segment in self.segments] + [self.end_point]
    for segment in self.segments:
      segment.plot(figax, **plot_options.fracture_segments)
    if plot_options.plot_fracture_points:
      for point in points:
        point.plot(figax, **plot_options.fracture_points)

    return figax
    
  def __repr__(self):
    points = [segment.point_1 for segment in self.segments] + [self.end_point]
    repr_ = 'Fracture on points :'
    for point in points:
      repr_ += ' {}'.format((point.x, point.y))
    return repr_


class No_Fracture:
  is_traversant = False
  def __repr__(self):
    return "No Fracture"


class Lazy_Fracture:
  def __init__(self, fracture):
    self.data = [(s.point_1.x, s.point_1.y) for s in fracture.segments] + [(fracture.end_point.x, fracture.end_point.y)]
    self.lengh = fracture.lengh
    self.is_traversant = fracture.is_traversant

  def to_full(self, mesh):
    points = [geo.Point(*p) for p in self.data]
    segments = [geo.Segment(p1, p2) for p1, p2 in zip(points, points[1:])]
    return Fracture(segments, mesh)

  def __repr__(self):
    repr_ = 'Fracture on points :'
    for d in self.data:
      x, y = d
      repr_ += ' {}'.format((x, y))
    return repr_


class Boundary_Mesh(geo.Polygon): # Should not inherit from Polygon : mesh might have holes
  """
  A boundary_mesh is a mesh of the boundary (i.e. in dim. 2, nodes and edges).
  #WARNING The edge.normal_vector should point to the interior
  """
  def __init__(self, edges, dirichlet_group, neumann_group):
    self.dirichlet_group = dirichlet_group
    self.neumann_group = neumann_group
    self.edges = edges
    self.dirichlet_parts = []
    self.neumann_parts = []
    self.nodes = [e.point_1 for e in self.edges]
    self.points = self.nodes
    
    # Regroup by parts
    dirichlet = []
    neumann = []
    for segment in self.edges:
      if self.is_edge_dirichlet(segment):
        dirichlet.append(segment)
      else:
        neumann.append(segment)
    
    for segments, parts in ((dirichlet, self.dirichlet_parts), (neumann, self.neumann_parts)):
      edge = segments.pop(0)
      temp_part = [edge]
      while segments:
        next_edge = segments.pop(0)
        if edge.point_2 is next_edge.point_1:
          temp_part.append(next_edge)
        else:
          parts.append(tuple(temp_part))
          temp_part = [next_edge]
        edge = next_edge
      
      if len(parts) > 0 and temp_part[-1].point_2 is parts[0][0].point_1:
          parts[0] = temp_part + parts[0]
      else:
        parts.append(temp_part)
    
    self._check_orientation()
    
    # Smallest rectangle with floe in it
    self.x_min = min(self.points, key=lambda p: p.x).x
    self.y_min = min(self.points, key=lambda p: p.y).y
    self.x_max = max(self.points, key=lambda p: p.x).x
    self.y_max = max(self.points, key=lambda p: p.y).y

  def _check_orientation(self):
    for parts in (self.dirichlet_parts, self.neumann_parts):
      for part in parts:
        edge = part[0]
        test_point = edge.mid_point + 0.1*edge.lengh/2*edge.normal_vector # Ugly, but works in use cases
        if not self.has_point(test_point):
          raise RuntimeError("Mesh file error : Boundary edges are not properly oriented.")

  def normal_vector(self, point, edge=None):
    edges = self.point_to_edges(point, edge)
    if len(edges) == 1:
      e, = edges
      if point.is_eq(e.point_1):
        edges.add(self.previous_edge(e))
      elif point.is_eq(e.point_2):
        edges.add(self.next_edge(e))
      else:
        return e.normal_vector
    
    assert len(edges) == 2
    e1, e2 = edges
    e = e1.normal_vector + e2.normal_vector
    return 1/np.linalg.norm(e)*e
  
  def inside_cone(self, point, edge=None):
    """
    If given, edge should have point
    """
    edges = self.point_to_edges(point, edge)
    
    if len(edges) == 1:
      e, = edges
      return geo.Cone(e.direction_vector, -e.direction_vector)
    elif len(edges) == 2:
      e1, e2 = edges
      return geo.Cone(e2.direction_vector, -e1.direction_vector)
  
  def is_edge_dirichlet(self, edge):
    if edge.group in self.dirichlet_group:
      return True
    return False
  
  def is_edge_neumann(self, edge):
    return not self.is_edge_dirichlet(edge)

  def is_node_boundary(self, node):
    if node in self.nodes:
      return True
    return False

  def is_node_dirichlet(self, node):
    """
    I.e. is the element on node an interior element or a boundary element.
    #INFO The node on the corner between dirichlet and neumann parts is dirichlet.
    """
    for e in node.of_edges:
      if self.is_edge_dirichlet(e):
        return True
    return False
  
  def is_node_neumann(self, node):
    return not self.is_node_dirichlet(node)
  
  def is_point_on_dirichlet_boundary(self, point):
    for part in self.dirichlet_parts:
      for edge in part:
        if edge.has_point(point):
          return True
    return False
    
  def boundary_neighboors(self, node):
    if not self.is_node_boundary(node):
      raise RuntimeError("Node {} is not on the boundary".format(node))
    boundary_neighboors = set()
    for n in node.neighboors:
      if self.is_node_boundary(n):
        boundary_neighboors.add(n)
    return boundary_neighboors

  def point_to_edge(self, point):
    """
    Returns the first edge with point in it.
    """
    for e in self.edges:
      if e.has_point(point):
        return e
    raise RuntimeError("Point {} is not on the boundary".format(point))

  def point_to_edges(self, point, edge=None):
    if edge:
      e1 = edge
    else:
      e1 = self.point_to_edge(point)
    e2, e3 = self.previous_edge(e1), self.next_edge(e1)
    if e3.has_point(point):
      return e1, e3
    elif e2.has_point(point):
      return e2, e1
    else:
      return e1,


class Mesh:
  """
  A Mesh object contains three important attributes :
  - a list of Nodes (indexed by their id number)
  - a set of Triangles
  For gmsh's mesh format : 2.2 0 8
  """
  def __init__(self, mesh_file, lines=None):
    self.nodes = []
    self.triangles = []
    self.boundary_edges = []
    self.interior_group = []
    
    if lines is None:
      with open(mesh_file) as meshtxt:
        lines = meshtxt.read()      

    for section in re.finditer(r'\$([a-zA-Z]*)\n(.*?)\n\$End\1', lines, flags=re.DOTALL):
      section_name = section.group(1)
      section_content = section.group(2)
      if section_name == "MeshFormat":
        self._read_mesh_format(section_content)
      elif section_name == "PhysicalNames":
        dirichlet_group, neumann_group = self._read_physical(section_content)
      elif section_name == "Nodes":
        self._read_nodes(section_content)
      elif section_name == "Elements":
        boundary_edges = self._read_elements(section_content)
      else:
        raise RuntimeError("Invalid Mesh File")
      
    self.boundary_mesh = Boundary_Mesh(boundary_edges, dirichlet_group, neumann_group)
    
    for t in self.triangles:
      for node in t.nodes:
        node._add_triangle(t.id_number)
        node._add_neighboors((n.id_number for n in t.nodes))
    
  def _read_mesh_format(self, lines):
    lines = lines.splitlines()
    mesh_format = lines[0].split(' ')[0]
    assert(mesh_format == '2.2')

  def _read_physical(self, lines):
    lines = lines.splitlines()
    number_lines = int(lines.pop(0))
    dirichlet_group, neumann_group = [], []
    assert(len(lines) == number_lines)
    for line in lines:
      line = line.split()
      if line[0] == '1':
        if 'D' in line[2]:
          dirichlet_group.append(int(line[1]))
        elif 'N' in line[2]:
          neumann_group.append(int(line[1]))
        else:
          neumann_group.append(int(line[1]))
          raise RuntimeWarning(".msh file has bad dirichlet/neumann boundary definition, see README file")
        
      elif line[0] == '2':
        self.interior_group.append(int(line[1]))
    return dirichlet_group, neumann_group

  def _read_nodes(self, lines):
    lines = lines.splitlines()
    number_lines = int(lines.pop(0))
    assert(len(lines) == number_lines)
    for i, line in enumerate(lines):
      e, x, y, *trash = line.split()
      x, y = float(x), float(y)
      current_node = Node(x, y, i, list_of_nodes=self.nodes, list_of_triangles=self.triangles)
      self.nodes.append(current_node)
  
  def _read_elements(self, lines):
    """
    Elements are edges and triangles in gmsh's language.
    Sort with dimension (edge dimension 1, triangle dimension 2).
    """
    lines_edges = re.findall(r'^[0-9]* 1 .*', lines, flags=re.MULTILINE)
    lines_triangles = re.findall(r'^[0-9]* 2 .*', lines, flags=re.MULTILINE)
    self._read_triangles(lines_triangles)
    return self._read_boundary_edges(lines_edges)

  def _read_boundary_edges(self, lines):
    boundary_edges = []
    for i, line in enumerate(lines):
      line = list(map(int, line.split()))
      node_1, node_2 = self.nodes[line[-2]-1], self.nodes[line[-1]-1]
      group = line[-4]
      edge = Edge(node_1, node_2, group)
      boundary_edges.append(edge)
      node_1._add_edge(edge)
      node_2._add_edge(edge)
    return boundary_edges

  def _read_triangles(self, lines):
    for i, line in enumerate(lines):
      line = list(map(int, line.split()))
      node_1, node_2, node_3 = line[-3:]
      node_1, node_2, node_3 = self.nodes[node_1-1], self.nodes[node_2-1], self.nodes[node_3-1] #a .mesh file starts indexing by 1.
      triangle = Triangle(node_1, node_2, node_3, i)
      self.triangles.append(triangle)

  def is_node_boundary(self, node):
    return self.boundary_mesh.is_node_boundary(node)
  
  def is_node_dirichlet(self, node):
    if node in self.boundary_mesh.nodes:
      return self.boundary_mesh.is_node_dirichlet(node)
    else:
      return False

  def is_node_neumann(self, node):
    if node in self.boundary_mesh.nodes:
      return self.boundary_mesh.is_node_neumann(node)
    else:
      return False
  
  def is_point_on_dirichlet_boundary(self, point):
    return self.boundary_mesh.is_point_on_dirichlet_boundary(point)

  def boundary_neighboors (self, node):
    return self.boundary_mesh.boundary_neighboors(node)
  
  def has_point(self, point, BoundaryException=False, with_boundary=False):
    return self.boundary_mesh.has_point(point, BoundaryException, with_boundary)

  def has_point_on_boundary(self, point):
    try:
     self.boundary_mesh.has_point(point, BoundaryException=True)
    except geo.OnBoundary:
      return True
    else:
      return False

  def plot(self, figax=None):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    ax.set_aspect('equal')
    
    # All this for no frame !
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(tight=True)
    ax.patch.set_alpha(0)
    
    for t in self.triangles:
      t.plot(figax, **plot_options.triangles)
    return figax


class Broken_Mesh(Mesh):
  """
  Defines a dictionnary local_refinement : each fractured_triangle t has two (sub)-dictionnarys:
  - on key -1 : the set of triangles obtained from the triangularization of the polygon from the left side of the fracture
  - on key  1 : the set of triangles ... from the right side of the fracture
  """
  def __init__ (self, fracture, mesh):
    self.fracture = fracture
    self.boundary_mesh = mesh.boundary_mesh
    self.nodes = mesh.nodes.copy()
    self.triangles = mesh.triangles.copy()
    for n in self.nodes:
      n._list_of_nodes = self.nodes
      n._list_of_triangles = self.triangles
    
    self.mid_nodes = set()
    self.tip_nodes = set()    
    self.local_refinement = {}
    self._fractured_triangles, self._clean_cut_triangles, self._one_node_on_fracture_triangles = set(), set(), set()
    self._triangle_to_fracture, self._triangle_to_fracture_large = {}, {}
    self._first_tip_triangles = set()
    self._last_tip_triangles = set()
    self._corner = []

    self._explore_mesh()
    self._check_admissibility()
    self._mesh_mid_refinement()

  def _get_init_triangles(self):
    # Find first fractured triangles
    if self.fracture.first_tip_enrichement:
      half_line = geo.Half_Line(self.fracture.first_segment.point_1, -self.fracture.first_segment.direction_vector)
      edges_inter = []
      for edge in self.boundary_mesh.edges:
        try:
          inter = geo.intersection(edge, half_line)
        except geo.OnBoundary as ex:
          if edge.point_1 in ex.intersection:
            inter = edge.point_1
          else:
            assert edge.point_2 in ex.intersection
            inter = edge.point_2
          edges_inter.append((edge, inter))
        except geo.GeometricException:
          pass
        else:
          edges_inter.append((edge, inter))
      
      edge = max(edges_inter, key=lambda x: x[1].array.dot(x[0].direction_vector))[0]
      
      to_explore = edge.point_1.of_triangles.intersection(edge.point_2.of_triangles)
      explored_triangles = set()
      while to_explore:
        to_explore_now = set()
        for t in to_explore:
          explored_triangles.add(t)
          
          if t.has_point(self.fracture.first_segment.point_1):
            self._first_tip_triangles.add(t)

          for edge in t.edges:
            try:
              geo.intersection(half_line, edge)
            except geo.NoIntersection:
              continue
            except geo.GeometricException:
              to_explore_now.update(tt for n in t.nodes for tt in n.of_triangles)
              break
            else:
              to_explore_now.update(tt for n in t.nodes for tt in n.of_triangles)
              break
          
        to_explore = to_explore_now - explored_triangles
      
      init_triangles = set()
      for t in self._first_tip_triangles:
        for n in t.nodes:
          for tt in n.of_triangles.difference(self._first_tip_triangles):
            for e in tt.edges:
              try:
                geo.intersection(e, self.fracture.segments[0])
              except geo.NoIntersection:
                pass
              except geo.GeometricException:
                init_triangles.add(tt)
              else:
                init_triangles.add(tt)
    else:
      boundary_edges = self.boundary_mesh.point_to_edges(self.fracture.start_point)
      init_triangles = set.union(*(n.of_triangles for e in boundary_edges for n in (e.point_1, e.point_2)))
    
    return init_triangles

  def _explore_mesh(self):
    triangles = self._get_init_triangles()
    for segment in self.fracture.segments[:-1]:
      triangles = self._explore_mesh_loc(triangles, segment)
      self._corner.append((triangles, [segment, self.fracture.next_segment(segment)]))
    end_triangles = self._explore_mesh_loc(triangles, self.fracture.last_segment)
    if self.fracture.last_tip_enrichement:
      self._last_tip_triangles = end_triangles
    self._fractured_triangles.difference_update(self._first_tip_triangles.union(self._last_tip_triangles))
    self._clean_cut_triangles.difference_update(self._first_tip_triangles.union(self._last_tip_triangles))
    self._one_node_on_fracture_triangles.difference_update(self._first_tip_triangles.union(self._last_tip_triangles))
    self.tip_nodes = set().union(n for t in self._first_tip_triangles.union(self._last_tip_triangles) for n in t.nodes)
  
  def _check_admissibility(self):
    # Checks for fracture admissibility
    # Forbid fracture too short
    if len(self._fractured_triangles.union(self._one_node_on_fracture_triangles).union(self._clean_cut_triangles)) < 2:
      raise NotAdmissibleFracture()
    # Forbid fracture that almost intersect
    for t in self._fractured_triangles:
      if len(self._triangle_to_fracture[t]) > 2:
        raise NotAdmissibleFracture()
      if len(self._triangle_to_fracture[t]) == 2:
        f1, f2 = self._triangle_to_fracture[t]
        if f1.point_2 is not f2.point_1 and f1.point_1 is not f2.point_2:
          raise NotAdmissibleFracture() 
    for t in self._clean_cut_triangles:
      if len(self._triangle_to_fracture_large[t]) > 2:
        raise NotAdmissibleFracture()
      elif len(self._triangle_to_fracture_large[t]) == 2:
        f1, f2 = self._triangle_to_fracture_large[t]
        if not geo.are_vectors_colinear(f1.direction_vector, f2.direction_vector):
          raise NotAdmissibleFracture()
    for t in self._last_tip_triangles:
      if t in self._fractured_triangles.union(self._clean_cut_triangles).union(self._one_node_on_fracture_triangles):
        if set(self._triangle_to_fracture_large[t]).difference(set([self.fracture.last_segment])):
          raise NotAdmissibleFracture()
      for n in t.nodes:
        for tt in n.of_triangles:
          if tt in self._triangle_to_fracture_large:
            if not set(self._triangle_to_fracture_large[tt]).intersection(set([self.fracture.last_segment])):
              raise NotAdmissibleFracture()
    # Forbid fractures almost to the boundary
    for t in self._last_tip_triangles.union(self._first_tip_triangles):
      if any(self.is_node_neumann(n) for n in t.points):
        raise NotAdmissibleFracture()

  def _explore_mesh_loc(self, to_explore, segment):
    """
    Explores the mesh recursively to determine where the fracture goes.
    """
    explored_triangles = set()
    end_triangles = set()
    States = Enum('States', 'FRACTURED CLEAN_CUT NO_INTERSECTION ONE_NODE_ON_FRACTURE')
    while to_explore:
      to_explore_now = set()
      for t in to_explore:
        explored_triangles.add(t)
        explore_neighboors = False
        
        if t.has_point(segment.point_2):
          end_triangles.add(t)

        # Intersections 
        intersections = []
        for edge in t.edges:
          try:
            intersection = geo.intersection(segment, edge)
          except geo.GeometricException as e:
            intersections.append(e)
          else:
            intersections.append(intersection)
        
        # Which intersection ?
        state = States.NO_INTERSECTION
        if any((type(i) is geo.Point for i in intersections)):
          state = States.FRACTURED
        elif any((type(i) is geo.OverlapError for i in intersections)):
          state = States.CLEAN_CUT
        elif any((type(i) is geo.OnBoundary for i in intersections)):
          relevant = [e for e in intersections if type(e) is geo.OnBoundary]
          if len(relevant) == 1:
            state = States.NO_INTERSECTION
          else:
            assert len(relevant) == 2
            e1, e2 = [edge for e in relevant for edge in e.boundary_of if edge in t.edges]
            if e1.point_1 is e2.point_2:
              e1, e2 = e2, e1
            assert e1.point_2 is e2.point_1
            cone = geo.Cone.acute(-e1.direction_vector, e2.direction_vector)
            if segment in relevant[0].boundary_of:
              assert segment in relevant[1].boundary_of
              if segment.point_1.is_eq(e1.point_2):
                if not cone.has_line(segment):
                  state = States.ONE_NODE_ON_FRACTURE
                elif cone.has_vector(segment.direction_vector):
                  state = States.FRACTURED
                else:
                  state = States.ONE_NODE_ON_FRACTURE
              else:
                assert segment.point_2.is_eq(e1.point_2)
                if not cone.has_line(segment):
                  state = States.ONE_NODE_ON_FRACTURE
                elif cone.has_vector(segment.direction_vector):
                  state = States.ONE_NODE_ON_FRACTURE
                else:
                  state = States.FRACTURED
            else:
              assert segment not in relevant[1].boundary_of
              if cone.has_line(segment): # has line
                state = States.FRACTURED
              else:
                state = States.ONE_NODE_ON_FRACTURE

        # Regular intersection
        if state is States.FRACTURED:
          explore_neighboors = True
          self.mid_nodes.update(t.nodes)
          self._fractured_triangles.add(t)
          if t in self._clean_cut_triangles:
            self._clean_cut_triangles.remove(t)
          elif t in self._one_node_on_fracture_triangles:
            self._one_node_on_fracture_triangles.remove(t)
          self._triangle_to_fracture.setdefault(t, []).append(segment)
          self._triangle_to_fracture_large.setdefault(t, []).append(segment)
        
        # Overlap intersection
        elif state is States.CLEAN_CUT:
          explore_neighboors = True
          intersection, = (i for i in intersections if type(i) is geo.OverlapError)
          edge = t.edges[intersections.index(intersection)]
          self.mid_nodes.update(set((edge.point_1, edge.point_2)))
          if t not in self._fractured_triangles:
            self._clean_cut_triangles.add(t)
          if t in self._one_node_on_fracture_triangles:
            self._one_node_on_fracture_triangles.remove(t)
          self._triangle_to_fracture_large.setdefault(t, []).append(segment)
        
        elif state is States.ONE_NODE_ON_FRACTURE:
          self._one_node_on_fracture_triangles.add(t)
          self._triangle_to_fracture_large.setdefault(t, []).append(segment)

        if explore_neighboors:
          neighboors_t = set.union(*(n.of_triangles for n in t.nodes)) - set((t,))
          to_explore_now.update(neighboors_t)
      
      to_explore_now.difference_update(explored_triangles)
      to_explore = to_explore_now
    
    return end_triangles
  
  def _break_triangle(self, t, fracture_segments):
    """
    #TODO
    Retrieve intersection data from explore_mesh
    """
    intersections = []
    for segment in fracture_segments: 
      for edge in t.edges:
        try:
          intersection = geo.intersection(edge, segment)
        except geo.NoIntersection:
          pass
        except geo.OverlapError as e:
          raise geo.OverlapError from e
        except geo.OnBoundary as e:
          intersection = next(iter(e.intersection))
          for i in intersections:
            if intersection.is_eq(i.point):
              if type(i) is T2:
                if edge not in i.edges:
                  if i.edges[0].point_2 is edge.point_1:
                    i.edges.append(edge)
                  else:
                    assert i.edges[0].point_1 is edge.point_2
                    i.edges.insert(0, edge)
              if type(i) is T3:
                if segment not in i.segments:
                  i.segments.append(segment)
              elif type(i) is T4:
                if segment not in i.segments:
                  i.segments.append(segment)
                if edge not in i.edges:
                  if i.edges[0].point_2 is edge.point_1:
                    i.edges.append(edge)
                  else:
                    assert i.edges[0].point_1 is edge.point_2
                    i.edges.insert(0, edge)
              break
          else:
            if edge in e.boundary_of and not segment in e.boundary_of:
              edge_point, = (point for point in (edge.point_1, edge.point_2) if point in e.intersection)
              intersections.append(T2(edges=[edge], segment=segment, point=edge_point))
            elif segment in e.boundary_of and not edge in e.boundary_of:
              intersections.append(T3(edge=edge, segments=[segment], point=next(iter(e.intersection))))
            else:
              edge_point, = (point for point in (edge.point_1, edge.point_2) if point in e.intersection)
              intersections.append(T4(edges=[edge], segments=[segment], point=edge_point))
        else:
          intersections.append(T1(edge=edge, segment=segment, point=intersection))
  
    if len(intersections) == 2:
      return self._break_triangle_two_intersections(t, fracture_segments, intersections)
    elif len(intersections) == 3: # Wrote this case a while ago, not sure it is actually used. Even if used, maybe we should render NonAdmissible fractures in this case.
      return self._break_triangle_three_intersections(t, fracture_segments, intersections)
    else:
      assert len(intersections) == 4 # Wrote this case a while ago, not sure it is actually used. Even if used, maybe we should render NonAdmissible fractures in this case.
      return self._break_triangle_four_intersections(t, fracture_segments, intersections)
  
  def _break_triangle_two_intersections(self, t, fracture_segments, intersections):
    """
    Regular fracture
    """
    # order intersections in the sense of direcion_vector 
    i1, i2 = intersections
    if len(fracture_segments) == 2:
      if type(i1) in (T3, T4):
        assert type(i2) in (T1, T2)
        if fracture_segments[0].has_point(i2.point, with_boundary=False):
          fracture_segments = fracture_segments[:-1]
        else:
          assert fracture_segments[1].has_point(i2.point, with_boundary=False)
          fracture_segments = fracture_segments[1:]
      elif type(i2) in (T3, T4):
        assert type(i1) in (T1, T2)
        if fracture_segments[0].has_point(i1.point, with_boundary=False):
          fracture_segments = fracture_segments[:-1]
        else:
          assert fracture_segments[1].has_point(i1.point, with_boundary=False)
          fracture_segments = fracture_segments[1:]
    
    segment_1 = fracture_segments[0]
    if len(fracture_segments) == 1 and geo.dist(i1.point, segment_1.point_1) > geo.dist(i2.point, segment_1.point_1):
      i1, i2 = i2, i1
    
    if len(fracture_segments) == 2:
      polygon_1 = [i1.point, segment_1.point_2, i2.point]
      polygon_2 = [i1.point, segment_1.point_2, i2.point]
    else:
      polygon_1 = [i1.point, i2.point]
      polygon_2 = [i1.point, i2.point]
    
    if type(i1) in (T2, T4) and type(i2) in (T2, T4):
      single_node, = set(t.nodes).intersection(set((i1.point, i2.point)))
      polygon_1.append(single_node)

    elif type(i1) in (T2, T4):
      if i2.edge in i1.edges:
        node_1, = set((i2.edge.point_1, i2.edge.point_2)).intersection(set([i1.point]))
        node_2, = set(t.nodes).intersection(set((node_1, i1.point)))
        polygon_1.append(node_1)
        polygon_1.append(node_2)
      else:
        node_1, node_2 = set(t.nodes).difference(set([i1.point]))
        polygon_1.append(node_1)
        polygon_2.append(node_2)

    elif type(i2) in (T2, T4):
      if i1.edge in i2.edges:
        node_2, = set((i1.edge.point_1, i1.edge.point_2)).difference(set([i2.point]))
        node_1, = set(t.nodes).intersection(set((node_2, i2.point)))
        polygon_1.append(node_1)
        polygon_1.append(node_2)
      else:
        node_1, node_2 = set(t.nodes).difference(set([i2.point]))
        polygon_1.append(node_1)
        polygon_2.append(node_2)

    elif i1.edge is i2.edge:
      node_1, node_3 = i1.edge.point_1, i1.edge.point_2
      if geo.dist(i2.point, node_1) > geo.dist(i2.point, node_3):
        node_1, node_3 = node_3, node_1
      node_2, = set(t.nodes).difference(set((node_1, node_3)))
      polygon_1.extend([node_1, node_2, node_3])
    
    else:
      single_node, = set((i1.edge.point_1, i1.edge.point_2)).intersection(set((i2.edge.point_1, i2.edge.point_2)))
      node_1 = set((i2.edge.point_1, i2.edge.point_2)).difference(set([single_node])).pop()    
      node_2 = set(t.nodes).difference(set((node_1, single_node))).pop()
      polygon_1.extend([node_1, node_2])
      polygon_2.append(single_node)

    # Polygon sides 
    if len(polygon_2) == 3:
      if segment_1.side(geo.Triangle(*polygon_2).mid_point) > 0:
        polygon_1, polygon_2 = polygon_2, polygon_1
    else:
      assert len(polygon_2) == 4
      inside_node = set(t.nodes).intersection(set(polygon_2)).difference(set(polygon_1)).pop()
      if segment_1.side(inside_node) > 0:
        polygon_1, polygon_2 = polygon_2, polygon_1
    
    return geo.Polygon(polygon_1).triangularize(), geo.Polygon(polygon_2).triangularize()
    
  def _break_triangle_three_intersections(self, t, fracture_segments, intersections):
    """
    Limit Case
    """
    i1, i2, i3 = intersections
    segment_1, segment_2 = fracture_segments
    if not segment_1.has_point(i2.point):
      i3, i2 = i2, i3
    elif not segment_2.has_point(i2.point):
      i1, i2 = i2, i1
    assert type(i2) in (T3, T4)
    
    # Construction of polygon_1
    polygon_1 = [i1.point, i2.point, i3.point]
   
    # Polygon_1 has four sides
    cone = geo.Marked_Cone.acute(-segment_1.direction_vector, segment_2.direction_vector, segment_1.point_2)
    if any((cone.has_point(node, with_boundary=False) for node in t.nodes)):
      assert type(i2) is T3
      inside_node, = (node for node in t.nodes if cone.has_point(node, with_boundary=False))
      polygon_1.append(inside_node)
      node_1 = (set([i1.edge.point_1, i1.edge.point_2]) - set([inside_node])).pop()
      node_2 = (set(t.nodes) - set([inside_node, node_1])).pop()
      polygon_2 = [i1.point, node_1, i2.point]
      polygon_3 = [i3.point, node_2, i2.point]
    
    # Node on fracture
    elif any((cone.has_point(node, with_boundary=True) for node in t.nodes)):
      assert type(i2) is T3
      node_1, node_2 = i2.edge.point_1. i2.edge.point_2
      if type(i1) in (T2, T4):
        if not set([node_2]).intersection(set((i3.edge.point_1, i3.edge.point_2))):
          node_1, node_2 = node_2, node_1
      else:
        assert type(i3) in (T2, T4)
        if not set([node_1]).intersection(set((i1.edge.point_1, i1.edge.point_2))):
          node_1, node_2 = node_2, node_1
      polygon_2 = [i1.point, node_1, i2.point]
      polygon_3 = [i3.point, node_2, i2.point]
    
    # Two intersections on same edge
    else:
      if type(i2) is T3:
        single_node = set((i2.edge.point_1, i2.edge.point_2)).intersection(set((i1.edge.point_1, i1.edge.point_2))).pop()
        node_1 = set((i1.edge.point_1, i1.edge.point_2)).intersection(set([single_node])).pop()
        node_2 = set((i2.edge.point_1, i2.edge.point_2)).intersection(set([single_node])).pop()
        if geo.dist(i1.point, single_node) > geo.dist(i3.point, single_node):
          polygon_2 = [i1.point, node_1, node_2, i2.point]
          polygon_3 = [i3.point, single_node, i2.point]
        else:
          polygon_2 = [i1.point, single_node, i2.point]
          polygon_3 = [i3.point, node_1, node_2, i2.point]
      else:
        e1, e2 = i2.edges
        single_node = set((e1.point_1, e1.point_2)).intersection(set((e2.point_1, e2.point_2))).pop()
        node_1, node_2 = set((e1.point_1, e1.point_2, e2.point_1, e2.point_2)).difference(set([single_node]))
        if geo.dist(node_1, i1.point) < geo.dist(node_1, i3.point):
          polygon_2 = [i1.point, i2.point, node_1]
          polygon_3 = [i3.point, i2.point, node_2]
        else:
          polygon_2 = [i1.point, i2.point, node_2]
          polygon_3 = [i3.point, i2.point, node_1]
    
    polygon_1, polygon_2, polygon_3 = geo.Polygon(polygon_1), geo.Polygon(polygon_2), geo.Polygon(polygon_3)
    # Side of polygon
    triangle = geo.Triangle(i1.point, i2.point, i3.point)
    if segment_1.side(triangle.mid_point) > 0:
      return polygon_1.triangularize(), polygon_2.triangularize().union(polygon_3.triangularize())
    else:
      return polygon_2.triangularize().union(polygon_3.triangularize()), polygon_1.triangularize()
  
  def _break_triangle_four_intersections(self, t, fracture_segments, intersections):
    """
    Case with 4 intersections
    """
    assert len(intersections) == 4
    i1, i2, i3, i4 = intersections
    segment_1, segment_2 = fracture_segments
    if geo.dist(i1.point, segment_1.point_1) > geo.dist(i2.point, segment_1.point_1):
      i1, i2 = i2, i1
    if geo.dist(i3.point, segment_2.point_1) > geo.dist(i4.point, segment_2.point_1):
      i3, i4 = i4, i3
    
    cone = geo.Marked_Cone.acute(-segment_1.direction_vector, segment_2.direction_vector, segment_1.point_2)
    
    # Inside intersection is pentagon
    if any((cone.has_point(point, with_boundary=False) for point in t.nodes)):
      inside_node, = (node for node in t.nodes if cone.has_point(node, with_boundary=False))
      node_1, node_2 = set(t.nodes) - set([inside_node])
      if i2.edge is i3.edge:
        polygon_1 = [i1.point, i2.point, i3.point, i4.point, inside_node]
        if geo.dist(i2.point, node_1) > geo.dist(i2.point, node_2):
          node_1, node_2 = node_2, node_1
      else:
        assert i1.edge is i4.edge
        polygon_1 = [i1.point, i2.point, inside_node, i3.point, i4.point]
        if geo.dist(i1.point, node_1) > geo.dist(i1.point, node_2):
          node_1, node_2 = node_2, node_1
      polygon_2 = [i1.point, node_1, i3.point]
      polygon_3 = [i2.point, node_2, i4.point]
    
    # One node is on fracture
    elif any((cone.has_point(point, with_boundary=True) for point in t.nodes)):
      polygon_1 = [i1.point, i2.point, i3.point, i4.point]
      if type(i1) in (T2, T4) or type(i4) in (T2, T4): 
        node_1, node_2 = set(t.nodes) - set([i1.point, i4.point])
        if geo.dist(i2.point, node_1) > geo.dist(i2.point, node_2):
          node_1, node_2 = node_2, node_1
      elif type(i2) in (T2, T4) or type(i3) in (T2, T4): 
        node_1, node_2 = set(t.nodes) - set([i2.point, i3.point])
        if geo.dist(i1.point, node_1) > geo.dist(i1.point, node_2):
          node_1, node_2 = node_2, node_1
      polygon_2 = [i1.point, node_1, i2.point]
      polygon_3 = [i4.point, node_2, i3.point]
    
    else:
      assert i2.edge is i3.edge and i1.edge is i4.edge
      polygon_1 = [i1.point, i2.point, i3.point, i4.point]
      single_node = set((i1.edge.point_1, i1.edge.point_2)).intersection(set((i2.edge.point_1, i2.edge.point_2))).pop()
      node_1 = (set((i1.edge.point_1, i1.edge.point_2)) - set([single_node])).pop()
      node_2 = (set((i2.edge.point_1, i2.edge.point_2)) - set([single_node])).pop()
      if geo.dist(i1.point, node_1) > geo.dist(i4.point, node_1):
        polygon_2 = [i1.point, single_node, i2.point]
        polygon_3 = [i4.point, node_1, node_2, i3.point]
      else:
        polygon_2 = [i1.point, node_1, node_2, i2.point]
        polygon_3 = [i4.point, single_node, i3.point]
    
    # Side of polygon
    polygon_1, polygon_2, polygon_3 = geo.Polygon(polygon_1), geo.Polygon(polygon_2), geo.Polygon(polygon_3)
    triangle = geo.Triangle(segment_1.point_2, i3.point, i4.point)
    if segment_1.side(triangle.mid_point) > 0:
      return polygon_1.triangularize(), polygon_2.triangularize().union(polygon_3.triangularize())
    else:
      return polygon_2.triangularize().union(polygon_3.triangularize()), polygon_1.triangularize()
  
  def _mesh_mid_refinement(self):
    # Refinement for mid_nodes
    for t in self._fractured_triangles:
      p1, p2 = self._break_triangle(t, self._triangle_to_fracture[t])
      self.local_refinement[t] = {1.0: p1, -1.0: p2}
      points_1 = set((point for p in p1 for point in p.points))
      points_2 = set((point for p in p2 for point in p.points))
      for n in set(t.nodes).intersection(points_1).difference(points_2):
        for tt in n.of_triangles:
          if tt not in self._clean_cut_triangles and tt not in self.local_refinement:
            self.local_refinement[tt] = {1.0: set([tt]), -1.0: set()}
      for n in set(t.nodes).intersection(points_2).difference(points_1):
        for tt in n.of_triangles:
          if tt not in self._clean_cut_triangles and tt not in self.local_refinement:
            self.local_refinement[tt] = {1.0: set(), -1.0: set([tt])}
    
    for t in self._clean_cut_triangles:
      segment_1 = self._triangle_to_fracture_large[t][0]
      side = segment_1.side(t.mid_point)
      self.local_refinement[t] = {side: set([t]), -side: set()}
    
    for t in self._one_node_on_fracture_triangles:
      if len(self._triangle_to_fracture_large[t]) == 1:
        side = self._triangle_to_fracture_large[t][0].side(t.mid_point)
      else:
        segment_1, segment_2 = self._triangle_to_fracture_large[t]
        cone = geo.Marked_Cone.acute(-segment_1.direction_vector, segment_2.direction_vector, segment_1.point_2)
        if cone.has_point(t.mid_point):
          side = 1.0
        else:
          side = -1.0
      self.local_refinement[t] = {side: set([t]), -side: set()}
  
    for c in self._corner:
      triangles = c[0]
      s1, s2 = c[1]
      for t in triangles:
        if not t in self.local_refinement:
          bn, = [n for n in t.nodes if n.is_eq(s1.point_2)]
          n1, n2 = set(t.nodes).difference(set([bn]))
          cone = geo.Marked_Cone(s2.direction_vector, -s1.direction_vector, bn)
          if cone.has_point(n1) and cone.has_point(n2):
            self.local_refinement[t] = {-1.0: set(), 1.0: set([t])}
          else:
            assert not cone.has_point(n1) and not cone.has_point(n2)
            self.local_refinement[t] = {-1.0: set([t]), 1.0: set()}
  
  def plot(self, figax=None):
    if not figax:
      figax = plt.subplots()
      fig, ax = figax
      ax.set_aspect('equal')
    
    super().plot(figax)
    
    t = next(iter(self.triangles))
    e = t.edges[0]
    plot_options.mid_nodes['markersize'] = plot_options.size_refinement_marker*e.lengh
    plot_options.tip_nodes['markersize'] = plot_options.size_refinement_marker*e.lengh

    if plot_options.plot_mid_nodes:
      for n in self.mid_nodes:
        n.plot(figax, **plot_options.mid_nodes)
    if plot_options.plot_tip_nodes:
      for n in self.tip_nodes:
        n.plot(figax, **plot_options.tip_nodes)
      
    if plot_options.plot_fractured_triangles:
      for t in self._fractured_triangles:
        for k in (-1, 1):
          for tt in self.local_refinement[t][k]:
            for edge in tt.edges:
              if not any((geo.are_vectors_colinear(edge.direction_vector, e.direction_vector) for e in t.edges)):
                edge.plot(figax, **plot_options.local_refinement_edges)
          mid_point = tt.mid_point
          if k == -1:
            mid_point.plot(figax, **plot_options.local_refinement_left)
          else:
            mid_point.plot(figax, **plot_options.local_refinement_right)
            
    self.fracture.plot(figax)
    
    return figax


class Broken_Mesh_Linear_Tip(Broken_Mesh):
  def __init__ (self, fracture, mesh):
    super().__init__(fracture, mesh)
    self._tip_enrichement = False
    self._old_and_new_tip_nodes = set()
    self._extra_tip_nodes = set()
    if not self.fracture.is_traversant:
      self._mesh_tip_refinement()
  
  def _mesh_tip_refinement(self):
    """
    Refinement of tip fracture without tip enrichement.
    """
    if set(n for t in self._first_tip_triangles for n in t.nodes).intersection(set(n for t in self._last_tip_triangles for n in t.nodes)):
      raise NotAdmissibleFracture
    if self.fracture.first_tip_enrichement:
      self.case_disparser(self._first_tip_triangles, self.fracture.first_segment, self.fracture.first_segment.point_1)
    if self.fracture.last_tip_enrichement: 
      self.case_disparser(self._last_tip_triangles, self.fracture.last_segment, self.fracture.last_segment.point_2)
  
  def case_disparser(self, tip_triangles, tip_segment, tip_point):
    if len(tip_triangles) == 1:
      t, = tip_triangles
      self._mesh_tip_case_1(tip_triangles, tip_segment, tip_point)
    
    elif len(tip_triangles) == 2:
      intersections = []
      t1, t2 = tip_triangles
      for t in (t1, t2):
        intersections.append([])
        for e in t.edges:
          try:
            inter = geo.intersection(tip_segment, e)
          except geo.GeometricException as inter:
            intersections[-1].append(inter)
          else:
            intersections[-1].append(inter)

      intersections_t1, intersections_t2 = intersections
      if any((type(i) == geo.OverlapError for inter in intersections for i in inter)):
        self._mesh_tip_case_2(tip_triangles, tip_segment, tip_point)
      else:
        if any((type(i) == geo.Point for i in intersections_t1)):
          tip_triangles.remove(t2)
        else:
          assert any((type(i) == geo.Point for i in intersections_t2))
          tip_triangles.remove(t1)
        self._mesh_tip_case_3(tip_triangles, tip_segment, tip_point)
    else:
      for e, t in [(e, t) for t in tip_triangles for e in t.edges]:
        try:
          inter = geo.intersection(tip_segment, e)
        except geo.GeometricException:
          pass
        else:
          break
      else:
        raise ShouldBeAdmissibleFracture
        # self._mesh_tip_case_2(tip_triangles, tip_segment, tip_point)
      tip_triangles = set([t])
      self._mesh_tip_case_3(tip_triangles, tip_segment, tip_point)
  
  def _mesh_tip_case_1(self, tip_triangles, tip_segment, tip_point):
    self._tip_enrichement = True
    tip_triangle, = tip_triangles
    p1, p2, p3 = tip_triangle.nodes
    tip_node = Node(tip_point.x, tip_point.y, len(self.nodes),
                of_triangles=set((tip_triangle.id_number, len(self.triangles), len(self.triangles) + 1)),
                neighboors=set((n.id_number for n in tip_triangle.nodes)), list_of_triangles = self.triangles, list_of_nodes=self.nodes)
    self.nodes.append(tip_node)
    self.mid_nodes.add(tip_node)
    self._extra_tip_nodes.add(tip_node)
    
    new_nodes = []
    for node in tip_triangle.nodes:
      n = Node(node.x, node.y, node.id_number, of_triangles=node._of_triangles.difference(set([tip_triangle.id_number])),
                        neighboors=node._neighboors.union(set([tip_node.id_number])), of_edges=node.of_edges,
                        list_of_triangles = self.triangles, list_of_nodes=self.nodes)
      new_nodes.append(n)
      self.nodes[n.id_number] = n
    n1, n2, n3 = new_nodes
    self._old_and_new_tip_nodes.add((p1, n1))
    self._old_and_new_tip_nodes.add((p2, n2))
    self._old_and_new_tip_nodes.add((p3, n3))
    for old, new in self._old_and_new_tip_nodes:
      try:
        self.mid_nodes.remove(old)
      except KeyError:
        pass
      else:
        self.mid_nodes.add(new)
    
    t1 = Triangle(n1, n2, tip_node, tip_triangle.id_number)
    t2 = Triangle(n1, n3, tip_node, len(self.triangles))
    t3 = Triangle(n2, n3, tip_node, len(self.triangles) + 1)
    self.triangles.extend([t2, t3])
    self.triangles[t1.id_number] = t1
    n1._add_triangle(t1.id_number), n1._add_triangle(t2.id_number) 
    n2._add_triangle(t1.id_number), n2._add_triangle(t3.id_number) 
    n3._add_triangle(t2.id_number), n3._add_triangle(t3.id_number) 
    
    old_triangles = set((t for n in (n1, n2, n3) for t in n.of_triangles)).difference(set((t1, t2, t3)))
    
    def to_new(a):
      for b in (n1, n2, n3):
        if a.id_number == b.id_number:
          return b
      return a
    
    for t in old_triangles:
      a1, a2, a3 = t.nodes
      a1 = to_new(a1)
      a2 = to_new(a2)
      a3 = to_new(a3)
      t_new = Triangle(a1, a2, a3, t.id_number)
      self.triangles[t_new.id_number] = t_new
      if t in self.local_refinement:
        self.local_refinement[t_new] = self.local_refinement[t]
        self.local_refinement.pop(t)

    if any((tip_segment.has_point(n) for n in tip_triangle.nodes)):
      self._mesh_tip_case_1_1(tip_segment, n1, n2, n3, t1, t2, t3, tip_node)
 
    else:
      self._mesh_tip_case_1_2(tip_segment, n1, n2, n3, t1, t2, t3)

  def _mesh_tip_case_1_1(self, tip_segment, n1, n2, n3, t1, t2, t3, tip_node):
    N, = [N for N in (n1, n2, n3) if tip_segment.has_point(N)]
    for t in (t1, t2, t3):
      if N in t.nodes:
        NN, = set(t.nodes).difference(set([N, tip_node]))
        side = tip_segment.side(NN)
        self.local_refinement[t] = {side: set([t]), -side: set()}
      else:
        self.local_refinement[t] = {1.0: set([t]), -1.0: set([t])}
    for n in (n1, n2, n3):
      for t in n.of_triangles:
        if t not in self.local_refinement:
          self.local_refinement[t] = {1.0: set([t]), -1.0: set([t])}
  
  def _mesh_tip_case_1_2(self, tip_segment, n1, n2, n3, t1, t2, t3):
    for N1, N2 in ((n1, n2), (n2, n3), (n1, n3)):
      try:
        geo.intersection(geo.Segment(N1, N2), tip_segment)
      except geo.NoIntersection:
        pass
      except geo.OnBoundary as e:
        assert tip_segment in e.boundary_of
        assert len(e.boundary_of) == 1
        break
      else:
        break

    for t in (t1, t2, t3):
      if N1 in t.nodes and N2 in t.nodes:
        p1, p2 = self._break_triangle(t, [tip_segment])
        self.local_refinement[t] = {1.0: p1, -1.0: p2}
      else:
        self.local_refinement[t] = {1.0: set([t]), -1.0: set([t])}
    
    for n in (n1, n2, n3):
      for t in n.of_triangles:
        if t not in self.local_refinement:
          self.local_refinement[t] = {1.0: set([t]), -1.0: set([t])}
  
  def _mesh_tip_case_2(self, tip_triangles, tip_segment, tip_point):
    raise NotImplementedError
    #t1, t2 = self._end_triangles
    #intersections = []

    #tip_node = Node(tip_point.x, tip_point.y, len(self.nodes),
    #           of_triangles=set((end_triangle.id_number, len(self.triangles), len(self.triangles) + 1)),
    #           neighboors=set((n.id_number for n in end_triangle.nodes)), list_of_triangles = self.triangles, list_of_nodes=self.nodes)
    #self.nodes.append(tip_node)
    #self.mid_nodes.add(tip_node)
    #self._tip_node = tip_node
    #
    #if any((type(ex) == geo.OverlapError for t, e, ex in intersections)):
    #  raise NotImplemented
    #else:
    #  for t, e, i in intersections:
    #    if type(i) is geo.Point:
    #      break
    #  for tt, e, ex in intersections:
    #    if tt is t and type(ex) is geo.OnBoundary:
    #      break
    #  t1, t2 = [t, set([t1, t2]).difference(set([t])).pop()]
    #  p1, p2 = self._break_triangle(t1, [self.fracture.last_segment])
    #  self.local_refinement[t1] = {1.0: p1, -1.0: p2}
    #  self.local_refinement[t2] = {1.0: set([t2]), -1.0: set([t2])}
    #  n1 = set((point for p in p1 for point in p.points)).intersection(set([e.point_1, e.point_2])).pop()
    #  n2 = set((point for p in p2 for point in p.points)).intersection(set([e.point_1, e.point_2])).pop()
    #  #for t in n1.of_triangles:
    #  #  if t not in self.local_refinement:
    #  #    self.local_refinement[t] = {1.0: set([t]), -1.0: set()}
    #  #for t in n2.of_triangles:
    #  #  if t not in self.local_refinement:
    #  #    self.local_refinement[t] = {1.0: set(), -1.0: set([t])}
    #  for t in n1.of_triangles.union(n2.of_triangles):
    #    if t not in self.local_refinement:
    #      self.local_refinement[t] = {1.0: set([t]), -1.0: set([t])}

  def _mesh_tip_case_3(self, tip_triangles, tip_segment, tip_point):
    t1, = tip_triangles
    self.mid_nodes.update(t1.nodes)
    p1, p2 = self._break_triangle(t1, [tip_segment])
    self.local_refinement[t1] = {1.0: p1, -1.0: p2}
    
    for n in t1.nodes:
      for t in n.of_triangles:
        if t not in self.local_refinement:
          self.local_refinement[t] = {1.0: set([t]), -1.0: set([t])}

