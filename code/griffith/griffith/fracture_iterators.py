import numpy as np
from math import pi, floor, cos, sin
import itertools

from . import geometry as geo
from . import mesh as msh


class Segments_From_Base_Point:
  def __init__(self, fracture_discretization, point, start_angle):
    self.point = point
    self.start_angle = start_angle
    self.angular_step = fracture_discretization.angular_step
    self.lengh_step = fracture_discretization.lengh_step
    self.min_lengh = fracture_discretization.min_lengh
    self.max_lengh = fracture_discretization.max_lengh
    
    self.angles = [self.start_angle]
    for k in range(1, floor(pi/self.angular_step)+1):
      self.angles.extend([self.start_angle - k*self.angular_step, self.start_angle + k*self.angular_step])
    
    self.angles_iterator = iter(self.angles.copy())
    self.lengh = self.min_lengh
  
  def __iter__(self):
    return self
  
  def __next__(self):
    try:
      self.angle = next(self.angles_iterator)
    except StopIteration:
      self.angles_iterator = iter(self.angles.copy())
      self.angle = next(self.angles_iterator)
      self.lengh += self.lengh_step
      if self.lengh > self.max_lengh:
        raise StopIteration
      
    return geo.Segment.polar(self.point, self.lengh, self.angle)
  
  def _remove_last_angle(self):
    self.angles.remove(self.angle)


class Segments_From_Center_Point:
  def __init__(self, fracture_discretization, point, start_angle):
    self.point = point
    self.start_angle = start_angle
    self.angular_step = fracture_discretization.angular_step
    self.lengh_step = fracture_discretization.lengh_step
    self.min_lengh = fracture_discretization.min_lengh
    self.max_lengh = fracture_discretization.max_lengh
    
    self.angles = [self.start_angle + k*self.angular_step for k in range(floor(pi/self.angular_step)+1)]
    self.angles_iterator = iter(self.angles)

    self.start_step, self.end_step = False, False
    self.start_lengh, self.end_lengh = 0, 0

  def __iter__(self):
    return self
  
  def __next__(self):
    if self.start_step:
      self.start_lengh += self.lengh_step/2
    if self.end_step:
      self.end_lengh += self.lengh_step/2
    
    if (self.start_lengh + self.end_lengh) > self.max_lengh or not (self.start_step or self.end_step):
      self.angle = next(self.angles_iterator)
      self.start_lengh, self.end_lengh = self.lengh_step/2, self.lengh_step/2
      self.start_step, self.end_step = True, True
    
    return self._create_segment()
    
  def stop_start_step(self):
    self.start_step = False
  
  def too_far_start(self):
    self.start_lengh -= self.lengh_step/2
    return self._create_segment()

  def stop_end_step(self):
    self.end_step = False
  
  def too_far_end(self):
    self.end_lengh -= self.lengh_step/2
    return self._create_segment()

  def _create_segment(self):
    base_point = self.point - (self.start_lengh/2*cos(self.angle), self.start_lengh/2*sin(self.angle))
    return geo.Segment.polar(base_point, self.start_lengh + self.end_lengh, self.angle)


class Admissible_Boundary_Point:
  """
  Returns boundary_point, normal_angle_oriented_interior at each iteration
  """
  def __init__(self, fracture_discretization, boundary_mesh):
    self.boundary_mesh = boundary_mesh
    self.boundary_step = fracture_discretization.boundary_step
    self.iter_segments = itertools.chain(*(p for p in boundary_mesh.neumann_parts))
  
    self.segment = next(self.iter_segments)
    self.point = self.segment.point_1
  def __iter__(self):
    return self
  
  def __next__(self):
    while True:
      d = self.boundary_step
      while True:
        h = geo.dist(self.point, self.segment.point_2)
        if d <= h:
          break
        else:
          self.segment = next(self.iter_segments)
          self.point = self.segment.point_1
          d = d - h
      self.point = self.point + d*self.segment.direction_vector
      
      if self.boundary_mesh.is_edge_neumann(self.segment):
        for p in (self.segment.point_1, self.segment.point_2):
          if self.point.is_eq(p):
            if self.boundary_mesh.is_node_dirichlet(p):
              break
        else:
          return self.point, self.segment
  
  def _is_admissible(self):
    #FIXME
    neighboring_edges = [self.boundary_mesh.next_edge(self.segment), self.boundary_mesh.previous_edge(self.segment)]
    for e in neighboring_edges:
      if self.boundary_mesh.is_edge_dirichlet(e):
        return False
    return True


class Admissible_Interior_Point:
  def __init__(self, fracture_discretization, boundary_mesh):
    self.boundary_mesh = boundary_mesh
    self.interior_step = fracture_discretization.interior_step
    x_min, x_max = boundary_mesh.x_min, boundary_mesh.x_max
    y_min, y_max = boundary_mesh.y_min, boundary_mesh.y_max
    h_nbr = floor((x_max - x_min)/self.interior_step)
    v_nbr = floor((y_max - y_min)/self.interior_step)
    self.corner_point = geo.Point(x_min, y_min)
    self.iter_lattice = itertools.product(range(1, h_nbr), range(1, v_nbr))
  
  def __iter__(self):
    return self
  
  def __next__(self):
    while True:
      point_lattice = np.array(next(self.iter_lattice))
      point = self.corner_point + self.interior_step*point_lattice
      if self.boundary_mesh.has_point(point, with_boundary=False):
        return point
      
    
class Admissible_Fractures_From_Fixed_Boundary_Point:
  def __init__(self, fracture_discretization, mesh, boundary_point=None, boundary_edge=None):
    self.mesh = mesh
    if boundary_point:
      self.boundary_point = boundary_point
    else:
      self.boundary_point = fracture_discretization.boundary_point
    assert self.boundary_point
    
    if boundary_edge:
      self.boundary_edge = boundary_edge
    else:
      self.boundary_edge = self.mesh.boundary_mesh.point_to_edge(self.boundary_point)
    
    self.inside_cone = self.mesh.boundary_mesh.inside_cone(self.boundary_point, self.boundary_edge) 
    normal_vector = self.mesh.boundary_mesh.normal_vector(self.boundary_point, self.boundary_edge)
    self.start_angle = geo.angle_to_ux(normal_vector)
    self.segment_iterator = Segments_From_Base_Point(fracture_discretization, self.boundary_point, self.start_angle)
    
  def __iter__(self):
    return self
  
  def __next__(self):
    while True:
      segment = next(self.segment_iterator)
      if self.inside_cone.has_vector(segment.direction_vector, with_boundary=False):
        fracture = msh.Fracture([segment], self.mesh)
        if not self.mesh.boundary_mesh.is_point_on_dirichlet_boundary(fracture.end_point):
          break
      self.segment_iterator._remove_last_angle()
      
    if fracture.is_traversant:
      self.segment_iterator._remove_last_angle()
    return fracture


class Admissible_Fractures_From_Boundary:
  def __init__(self, fracture_discretization, mesh):
    self.mesh = mesh
    self.fracture_discretization = fracture_discretization
    self.boundary_point_iterator = Admissible_Boundary_Point(fracture_discretization, mesh.boundary_mesh)
    self.fracture_iterator_fixed_bp = iter([])

  def __iter__(self):
    return self

  def __next__(self):
    try:
      fracture = next(self.fracture_iterator_fixed_bp)
    except StopIteration:
      self.boundary_point, self.boundary_edge = next(self.boundary_point_iterator)
      self.fracture_iterator_fixed_bp = Admissible_Fractures_From_Fixed_Boundary_Point(self.fracture_discretization, self.mesh, self.boundary_point, self.boundary_edge) 
      fracture = next(self.fracture_iterator_fixed_bp)
    return fracture 


class Admissible_Fractures_From_Interior:
  def __init__(self, fracture_discretization, mesh):
    self.mesh = mesh
    self.fracture_discretization = fracture_discretization
    self.start_angle = fracture_discretization.interior_start_angle
    self.interior_point_iterator = Admissible_Interior_Point(fracture_discretization, mesh.boundary_mesh)
    self.segment_iterator = iter([])

  def __iter__(self):
    return self
  
  def __next__(self):
    try:
      segment = next(self.segment_iterator)
    except StopIteration:
      interior_point = next(self.interior_point_iterator)
      self.segment_iterator = Segments_From_Center_Point(self.fracture_discretization, interior_point, start_angle=self.start_angle)
      if self.fracture_discretization.interior_fast_init:
        self.segment_iterator.max_lengh = self.segment_iterator.min_lengh
      segment = next(self.segment_iterator)
    
    fracture = msh.Fracture([segment], self.mesh)
    if not fracture.first_tip_enrichement:
    #if self.mesh.boundary_mesh.is_point_on_boundary(fracture.start_point): # Looks like #fracture
      self.segment_iterator.stop_start_step()
      if self.mesh.boundary_mesh.is_point_on_dirichlet_boundary(fracture.start_point):
        segment = self.segment_iterator.too_far_start()
        fracture = msh.Fracture([segment], self.mesh)
    
    if not fracture.last_tip_enrichement:
    #if self.mesh.boundary_mesh.is_point_on_boundary(fracture.end_point):
      self.segment_iterator.stop_end_step()
      if self.mesh.boundary_mesh.is_point_on_dirichlet_boundary(fracture.end_point):
        segment = self.segment_iterator.too_far_end()
        fracture = msh.Fracture([segment], self.mesh)
    
    return fracture


class Admissible_Segments_From_Tip:
  def __init__(self, fracture_discretization, mesh, fracture, tip):
    self.mesh = mesh
    self.fracture = fracture
    self.tip = tip
    
    if tip is self.fracture.end_point:
      self.start_angle = geo.angle_to_ux(fracture.last_segment.direction_vector)
      self.segment_iterator = Segments_From_Base_Point(fracture_discretization, fracture.last_segment.point_2, self.start_angle)
      self.swap = False
    else:
      assert tip is self.fracture.start_point
      self.start_angle = geo.angle_to_ux(-fracture.first_segment.direction_vector)
      self.segment_iterator = Segments_From_Base_Point(fracture_discretization, fracture.first_segment.point_1, self.start_angle)
      self.swap = True

  def __iter__(self):
    if self.tip is self.fracture.end_point and not self.fracture.last_tip_enrichement:
      return iter([])
    elif self.tip is self.fracture.start_point and not self.fracture.first_tip_enrichement:
      return iter([])
    else:
      return self
  
  def __next__(self):
    while True:
      segment = next(self.segment_iterator)
      if self.swap:
        segment._change_orientation()
      fracture = msh.Fracture([segment], self.mesh)
      if self._is_admissible(segment, fracture):
        break
    
    return segment

  def _is_admissible(self, segment, fracture):
    if self.tip is self.fracture.end_point:
      if geo.are_vectors_colinear(segment.direction_vector, self.fracture.last_segment.direction_vector) and np.dot(segment.direction_vector, self.fracture.last_segment.direction_vector) < 0:
        self.segment_iterator._remove_last_angle()
        return False
    else:
      if geo.are_vectors_colinear(segment.direction_vector, self.fracture.first_segment.direction_vector) and np.dot(segment.direction_vector, self.fracture.first_segment.direction_vector) < 0:
        self.segment_iterator._remove_last_angle()
        return False
    
    if self.tip is self.fracture.end_point:
      segments = self.fracture.segments[:-2]
    else:
      segments = self.fracture.segments[1:]

    for s in segments:
      try:
        geo.intersection(segment, s)
      except geo.NoIntersection:
        pass
      except geo.GeometricException:
        self.segment_iterator._remove_last_angle()
        return False
      else:
        self.segment_iterator._remove_last_angle()
        return False
    
    if self.tip is self.fracture.end_point:
      if fracture.last_tip_enrichement:
        return True
      else:
        self.segment_iterator._remove_last_angle()
        if not self.mesh.boundary_mesh.is_point_on_dirichlet_boundary(fracture.end_point):
          return True
        else:
          return False
    else:
      if fracture.first_tip_enrichement:
        return True
      else:
        self.segment_iterator._remove_last_angle()
        if not self.mesh.boundary_mesh.is_point_on_dirichlet_boundary(fracture.start_point):
          return True
        else:
          return False
        

class Admissible_Fractures_From_Fracture:
  def __init__(self, fracture_discretization, mesh, fracture):
    self.mesh = mesh
    self.fracture = fracture
    first_segment_iter = itertools.chain(Admissible_Segments_From_Tip(fracture_discretization, mesh, fracture, fracture.start_point), [None])
    last_segment_iter = itertools.chain(Admissible_Segments_From_Tip(fracture_discretization, mesh, fracture, fracture.end_point), [None])
    self.segments_iterator = itertools.product(first_segment_iter, last_segment_iter)
  
  def __iter__(self):
    if self.fracture.is_traversant:
      return iter([])
    else:
      return self
  
  def __next__(self):
    while True:
      s1, s2 = next(self.segments_iterator)
      segments = self.fracture.segments.copy()
      if s1 is None and s2 is None:
        continue
      
      if s1:
        segments.insert(0, s1)
      if s2:
        segments.append(s2)
      
      if s1 and s2:
        try:
          geo.intersection(s1, s2)
        except geo.NoIntersection:
          break
        except geo.GeometricException:
          pass
      else:
        break

    return msh.Fracture(segments, self.mesh)


class Admissible_Fractures_From_Fracture_Fast:
  def __init__(self, fracture_discretization, mesh, fracture):
    self.mesh = mesh
    self.fracture = fracture
    self.first_segment_iter = iter(Admissible_Segments_From_Tip(fracture_discretization, mesh, fracture, fracture.start_point))
    self.last_segment_iter = iter(Admissible_Segments_From_Tip(fracture_discretization, mesh, fracture, fracture.end_point))

  def __iter__(self):
    if self.fracture.is_traversant:
      return iter([])
    else:
      return self
  
  def __next__(self):
    segments = self.fracture.segments.copy()
    try:
      s = next(self.first_segment_iter)
    except StopIteration:
      s = next(self.last_segment_iter)
      segments.append(s)
    else:
      segments.insert(0, s)

    return msh.Fracture(segments, self.mesh)
