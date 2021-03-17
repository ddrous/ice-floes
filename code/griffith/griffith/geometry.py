import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, cos, sin, fabs, atan2

PRECISION = 1e-5


class GeometricException(Exception):
  pass


class OverlapError(GeometricException):
  pass
  """
  Thrown out in case of two overlaping lines in intersection().
  """

class LineError(GeometricException):
  """
  Line with one point Error
  """
  pass


class ConeError(GeometricException):
  """
  Cone with same two vectors error
  """


class NoIntersection(GeometricException):
  pass


class OnBoundary(GeometricException):
  def __init__(self, intersection=set(), boundary_of=set()):
    self.intersection = intersection
    self.boundary_of = boundary_of


def intersection(line_1, line_2):
  """
  Works with subclasses of Line
  Raises NoIntersection if no intersection
  Raises OverlapError if Overlap
  Raises OnBoundary if intersection on boundary
  Returns Point if single point intersection
  WARNING: NOT SYMETRIC !
  """
  A = np.vstack((line_1.normal_vector, line_2.normal_vector))
  
  if fabs(np.linalg.det (A)) < PRECISION:
    direction_vector = line_1.direction_vector
    normal_vector = line_1.normal_vector
    # We check if the "true" lines are overlapping or not
    if fabs(np.dot(normal_vector, line_1.point_1.array) - np.dot(normal_vector, line_2.point_1.array)) > PRECISION:
      raise NoIntersection()
    # Two lines
    elif type(line_1) is Line or type(line_2) is Line:
      raise OverlapError
    # Two half-lines
    elif type(line_1) is Half_Line and type(line_2) is Half_Line:
      if np.dot(line_1.direction_vector, line_2.direction_vector) > 0.5:
        raise OverlapError
      elif line_2.point.is_eq(line_1.point):
        raise OnBoundary(intersection=set((line_2.point, line_1.point)), boundary_of=set((line_1, line_2)))
      elif line_2.has_point(line_1.point):
        raise OverlapError
      else:
        raise NoIntersection()

    # Otherwise, we swap orientation to make the computation more readable
    if np.dot(line_1.direction_vector, line_2.direction_vector) < 0:
      try:
        line_1 = line_1.swap_orientation()
      except NotImplementedError:
        line_2 = line_2.swap_orientation()
      else:
        direction_vector = line_1.direction_vector
        normal_vector = line_1.normal_vector

    # Two segments
    if isinstance(line_1, Segment) and isinstance(line_2, Segment):
      # We only look at the case in wich line_2 is the highest (in the sense of direction_vector)
      if np.dot(direction_vector, line_1.point_2.array) > np.dot(direction_vector, line_2.point_2.array) + PRECISION:
        line_1, line_2 = line_2, line_1
      ps1 = np.dot(direction_vector, line_1.point_1.array)
      ps2 = np.dot(direction_vector, line_1.point_2.array)
      ps3 = np.dot(direction_vector, line_2.point_1.array)
      ps4 = np.dot(direction_vector, line_2.point_2.array)
      if (ps3 - ps1) > PRECISION:
        if ps2 > ps3 + PRECISION:
          raise OverlapError
        elif ps2 + PRECISION < ps3:
          raise NoIntersection()
        else:
          assert fabs(ps3-ps2) < PRECISION
          raise OnBoundary(intersection=set((line_2.point_1, line_1.point_2)), boundary_of=set((line_1, line_2)))
      else:
        raise OverlapError
    
    # One segment and one half-line
    if isinstance(line_1, Segment) and type(line_2) is Half_Line:
      line_1, line_2 = line_2, line_1
    if type(line_1) is Half_Line and isinstance(line_2, Segment):
      ps = np.dot(line_2.point_2.array - line_1.point.array, direction_vector)
      if ps < -PRECISION:
        raise NoIntersection()
      elif ps > PRECISION:
        raise OverlapError
      else:
        raise OnBoundary(intersection=set((line_1.point, line_2.point_2)), boundary_of=set((line_1, line_2)))
    else: # We never know ...
      raise RuntimeError("Could not determine intersection")
  
  intersection = Point(*np.linalg.solve (A, np.array ([np.dot(line_1.point_1.array, line_1.normal_vector), np.dot(line_2.point_2.array, line_2.normal_vector)])))

  err_intersection, boundary_of = set(), set()
  for p in (line_1.point_1, line_1.point_2):
    if p.is_eq(intersection):
      err_intersection.add(p)
      boundary_of.add(line_1)
  for p in (line_2.point_1, line_2.point_2):
    if p.is_eq(intersection):
      err_intersection.add(p)
      boundary_of.add(line_2)

  if not line_1.has_point(intersection):
    raise NoIntersection()
  elif not line_2.has_point(intersection):
    raise NoIntersection()
  elif err_intersection:
    raise OnBoundary(intersection=err_intersection, boundary_of=boundary_of)
  else:
    return intersection


def dist(point_1, point_2):
  return sqrt((point_1.x - point_2.x)**2 + (point_1.y - point_2.y)**2)


def angle(vector_1, vector_2):
  return atan2(vector_2[1], vector_2[0]) - atan2(vector_1[1], vector_1[0])


def angle_to_ux(vector):
  return angle(np.array([1, 0]), vector)


def rotation(base, point, angle):
  reduced = point - base
  rotation_matrix = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
  
  return base + rotation_matrix.dot(reduced.array)

def are_vectors_colinear(vector_1, vector_2):
  A = np.vstack((vector_1, vector_2))
  
  if np.abs(np.linalg.det (A)) < PRECISION:
    return True
  else:
    return False


class Cone:
  def __init__(self, vector_1, vector_2):
    self.vector_1, self.vector_2 = vector_1, vector_2
    self.angle = angle(self.vector_1, self.vector_2)
    if fabs(self.angle) < PRECISION:
      raise ConeError()
    elif self.angle < 0:
      self.angle += 2*pi
  
  @classmethod
  def acute(cls, vector_1, vector_2):
    obj = cls(vector_1, vector_2)
    if obj.angle > pi:
      obj.angle = 2*pi - obj.angle
      obj.vector_1, obj.vector_2 = obj.vector_2, obj.vector_1
    return obj
  
  def has_vector(self, vector, BoundaryException=False, with_boundary=True):
    angle_int = angle(self.vector_1, vector)
    if fabs(angle_int) < PRECISION:
      if BoundaryException:
        raise OnBoundary(boundary_of=set(self.vector_1))
      else:
        return with_boundary

    if angle_int < -PRECISION:
      angle_int += 2*pi
    
    if fabs(angle_int-self.angle) < PRECISION:
      if BoundaryException:
        raise OnBoundary(boundary_of=set(self.vector_1))
      else:
        return with_boundary

    if angle_int > self.angle + PRECISION:
      return False
    else:
      return True

  def has_line(self, line, BoundaryException=False, with_boundary=True):
    if self.has_vector(line.direction_vector):
      return True
    elif self.has_vector(-line.direction_vector):
      return True
    else:
      return False


class Marked_Cone(Cone):
  def __init__(self, vector_1, vector_2, base_point):
    super().__init__(vector_1, vector_2)
    self.base_point = base_point

  def has_point(self, point, BoundaryException=False, with_boundary=True):
    vector = (point - self.base_point).array
    return super().has_vector(vector, BoundaryException, with_boundary)

  @classmethod
  def acute(cls, vector_1, vector_2, base_point):
    obj = cls(vector_1, vector_2, base_point)
    if obj.angle > pi:
      obj.angle = 2*pi - obj.angle
      obj.vector_1, obj.vector_2 = obj.vector_2, obj.vector_1
    return obj


class Line:
  def __init__(self, point_1, point_2):
    if point_1.is_eq(point_2):
      raise LineError
    self.point_1 = point_1
    self.point_2 = point_2
    self.direction_vector = np.array([point_2.x - point_1.x, point_2.y - point_1.y])
    self.normal_vector = np.array ([-self.direction_vector [1], self.direction_vector[0]])
    norm = sqrt((self.direction_vector[0])**2 + (self.direction_vector[1])**2) # math.sqrt faster than numpy.linalg.norm
    self.direction_vector *= 1/norm
    self.normal_vector *= 1/norm
    
  def swap_orientation(self):
    return Line(self.point_2, self.point_1)
  
  def has_point(self, point, *args, **kwargs):
    """
    *args, **kwargs here to have same signature as Segments
    """
    if np.abs(np.dot(self.normal_vector, np.array([point.x - self.point_1.x, point.y - self.point_1.y]))) <  PRECISION:
      return True
    return False
 
  def side(self, point):
    """
    On which side of the half plan is this point ?
    """
    result = np.dot (point.array - self.point_1.array, self.normal_vector)
    if np.abs (result) < PRECISION:
      return 0
    else:
      return np.sign (result)

  def __repr__(self):
    return "Line goes through {} and {}".format(self.point_1, self.point_2)


class Segment(Line):
  """
  Thoses are oriented segment.
  Point_1 should be the start of the fracture, and point_2 the end.
  (direction_vector, normal_vector) is a direct orthonormed basis.
  """
  def __init__(self, point_1, point_2):
    super().__init__(point_1, point_2)

  @classmethod
  def polar(cls, base_point, r, theta):
    point_1 = base_point
    point_2 = point_1 + (r*cos(theta), r*sin(theta))
    return cls(point_1, point_2)
 
  @property
  def mid_point(self):
    try:
      self._mid_point
    except AttributeError:
      p1, p2 = self.point_1, self.point_2
      self._mid_point = Point(0.5*(p1.x + p2.x), 0.5*(p1.y + p2.y))
    finally:
      return self._mid_point

  def swap_orientation(self):
    return Segment(self.point_2, self.point_1)
  
  def has_point(self, point, BoundaryException=False, with_boundary=True):
    dist_1 = dist(self.point_1, point)
    dist_2 = dist(self.point_2, point)
    if dist_1 < PRECISION:
      if BoundaryException:
        raise OnBoundary(intersection=set((self.point_1, point)))
      else:
        return with_boundary
    elif dist_2 < PRECISION:
      if BoundaryException:
        raise OnBoundary(intersection=set((self.point_2, point)))
      else:
        return with_boundary
    elif dist_1 < self.lengh + PRECISION and dist_2 < self.lengh + PRECISION and super().has_point(point):
      return True
    else:
      return False
  
  def _change_orientation(self):
    self.point_1, self.point_2 = self.point_2, self.point_1
    self.direction_vector = -self.direction_vector
    self.normal_vector = -self.normal_vector
  
  @property
  def lengh(self):
    try:
      self._lengh
    except AttributeError:
      self._lengh = dist(self.point_1, self.point_2)
    finally:
      return self._lengh

  def __repr__(self):
    return "Segment between {} and {}".format(self.point_1, self.point_2)
    
  def plot(self, figax=None, **kwargs):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    x = [self.point_1.x, self.point_2.x]
    y = [self.point_1.y, self.point_2.y]
    ax.plot(x, y, **kwargs)
    return figax


class Half_Line(Line):
  def __init__(self, point, vector):
    super().__init__(point, point+vector)
    self.point = point
    self.vector = vector
    
  def has_point(self, point, BoundaryException=False, with_boundary=True):
    dist_1 = dist(point, self.point)
    if dist_1 < PRECISION:
      if BoundaryException:
        raise OnBoundary(intersection=((self.point, point)))
      else:
        return with_boundary
    elif super().has_point(point):
      if np.dot((point - self.point).array, self.vector) > 0:
        return True
    return False

  def __repr__(self):
    return "Half_Line starts from {} in direction {}".format(self.point, self.vector)
  
  def swap_orientation(self):
    raise NotImplementedError


class Polyline:
  def __init__ (self, segments):
    self.segments = segments
      
  @property
  def lengh(self):
    try:
      self._lengh
    except AttributeError:
      self._lengh = 0
      for s in self.segments:
        self._lengh += s.lengh
    finally:
      return self._lengh
  
  @property
  def last_segment(self):
    return self.segments[-1]
  
  @last_segment.setter
  def last_segment(self, last_segment):
    self.segments[-1] = last_segment
  
  @property
  def end_point(self):
    return self.last_segment.point_2

  @property
  def start_point(self):
    return self.first_segment.point_1

  @property
  def first_segment(self):
    return self.segments[0]

  @first_segment.setter
  def first_segment(self, first_segment):
    self.segments[0] = first_segment
  
  def has_point(self, point, BoundaryException=False, with_boundary=True):
    if point.is_eq(self.first_segment.point_1):
      if BoundaryException:
        raise OnBoundary(intersection=set((point, self.first_segment.point_1)))
      else:
        return with_boundary
    elif point.is_eq(self.last_segment.point_2):
      if BoundaryException:
        raise OnBoundary(intersection=set((point, self.last_segment.point_2)))
      else:
        return with_boundary
    else:
      for s in self.segments:
        value = s.has_point(point)
        if value:
          return True
      return False
  
  def next_segment(self, s):
    return self.segments[(self.segments.index(s) + 1)]
  
  def __repr__ (self):
    str = "Polyline made of following segments :"
    for s in self.segments:
      str += s.__repr__()
    return str
  
  def plot(self, figax=None, **kwargs):
    if not figax:
      figax = plt.subplots()
    for s in self.segments:
      s.plot(figax, **kwargs)
    return figax
        
  def curvilinear_abscissa(self, x):
    assert 0 <= x <= self.lengh
    now = 0
    for s in self.segments:
      if now + s.lengh > x:
        return s.point_1 + (x-now)*s.direction_vector
      else:
        now += s.lengh


class Point:
  def __init__(self, x, y):
    self.x = np.float(x)
    self.y = np.float(y)
  
  def __add__(self, vector):
    if isinstance(vector, (list, tuple, np.ndarray)):
      return Point(self.x + vector[0], self.y + vector[1])
    elif isinstance(vector, Point):
      return Point(self.x + vector.x, self.y + vector.y)
    else:
      raise NotImplementedError

  def __sub__(self, vector):
    if isinstance(vector, (list, tuple, np.ndarray)):
      return Point(self.x - vector[0], self.y - vector[1])
    elif isinstance(vector, Point):
      return Point(self.x - vector.x, self.y - vector.y)
    else:
      raise NotImplementedError
  
  @property
  def array(self):
    return np.array([self.x, self.y])
  
  def is_eq(self, other):
    if fabs(self.x-other.x) < PRECISION and fabs(self.y-other.y) < PRECISION:
      return True
    return False

  def __repr__ (self) :
    return "Point at : (x, y) = {}".format((self.x, self.y))
  
  def plot(self, figax=None, **kwargs):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    ax.plot(self.x, self.y, **kwargs)
    return figax


class Triangle:
  def __init__(self, point_1, point_2, point_3):
    self.points = (point_1, point_2, point_3)
    self.edges = (Segment(point_1, point_2), Segment(point_2, point_3), Segment(point_3, point_1))
    
  def __repr__(self):
    return "Triangle on points {}".format(self.points)
  
  def next_edge(self, e):
    index = self.edges.index(e)
    try:
      edge = self.edges[index + 1]
    except IndexError:
      edge = self.edges[0]
    return edge

  def previous_edge(self, e):
    return self.edges[self.edges.index(e) - 1]
  
  @property
  def mid_point(self):
    try:
      self._mid_point
    except AttributeError:
      p3 = self.points[2]
      s1 = self.edges[0]
      self._mid_point = Segment(s1.mid_point, p3).mid_point
    finally:
      return self._mid_point

  def has_point(self, point, BoundaryException=False, with_boundary=True):
    s1, s2, s3 = self.edges
    p1, p2, p3 = self.points
    ps1 = np.dot(s1.normal_vector, (point - p1).array)
    ps2 = np.dot(s2.normal_vector, (point - p2).array)
    ps3 = np.dot(s3.normal_vector, (point - p3).array)
    
    for ss, ps in ((s1, ps1), (s2, ps2), (s3, ps3)):
      if fabs(ps) < PRECISION and ss.has_point(point):
        if BoundaryException:
          inter = set([point])
          boundary_of = set([ss])
          if point.is_eq(ss.point_2):
            inter.add(ss.point_2)
            boundary_of.add(self.next_edge(ss))
          elif point.is_eq(ss.point_1):
            inter.add(ss.point_1)
            boundary_of.add(self.previous_edge(ss))
          raise OnBoundary(intersection=inter, boundary_of=boundary_of)
        else:
          return with_boundary

    if np.sign(ps1) == np.sign(ps2) and np.sign(ps1) == np.sign(ps3):
      return True
    else:
      return False
  
  def other_points(self, p):
    if p in self.points:
      return set(self.points).difference((p,))
    for pp in self.points:
      if pp.is_eq(p):
        return set(self.points).difference((pp,))
    raise RuntimeError("Point {} not vertex of triangle {}".format(p, self))
  
  @property
  def area(self):
    try:
      self._area
    except AttributeError:
      p1, p2, p3 = self.points
      self._area = np.abs(np.linalg.det (np.array ([[p1.x, p1.y, 1],[p2.x, p2.y, 1],[p3.x, p3.y, 1]])))
    finally:
      return self._area
  
  def plot(self, figax=None, fill=False, color='k', **kwargs):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    p = [p for p in self.points]
    p.append(p[0])
    x = [p_.x for p_ in p]
    y = [p_.y for p_ in p]

    ax.plot(x, y, color=color, **kwargs)
    return figax
    

class Circle:
  def __init__(self, center, radius):
    self.center = center
    self.radius = radius

  def has_point(self, p, BoundaryException=False, with_boundary=True):
    if dist(p, self.center) < self.radius - PRECISION:
      return True
    elif dist(p, self.center) < self.radius + PRECISION:
      if BoundaryException:
        raise OnBoundary()
      else:
        return with_boundary
    else:
      return False
    
  def plot(self, figax=None, **kwargs):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax  
    ax.add_artist(plt.Circle((self.center.x, self.center.y), self.radius, **kwargs))
    return figax


class Polygon:
  def __init__ (self, points):
    self.points = points
    self.lengh = len(points)
    if self.lengh < 3:
      raise ValueError("Not a polygon")
    self.edges = [Segment(points[i], points[(i+1) % self.lengh]) for i in range(self.lengh)]
  
  def next_edge(self, e):
    index = self.edges.index(e)
    try:
      edge = self.edges[index + 1]
    except IndexError:
      edge = self.edges[0]
    return edge

  def previous_edge(self, e):
    return self.edges[self.edges.index(e) - 1]

  def has_point(self, point, BoundaryException=False, with_boundary=False):
    half_line = Half_Line(point, np.array([1, 1]))
    # Boundary
    if BoundaryException or with_boundary:
      for edge in self.edges:
        if edge.has_point(point):
          if BoundaryException:
            inter = set([point])
            boundary_of = set([edge])
            if point.is_eq(edge.point_2):
              inter.add(edge.point_2)
              boundary_of.add(self.next_edge(edge))
            elif point.is_eq(edge.point_1):
              inter.add(edge.point_1)
              boundary_of.add(self.previous_edge(edge))
            raise OnBoundary(intersection=inter, boundary_of=boundary_of)
          else:
            return with_boundary
    elif not with_boundary: # Safety
      for edge in self.edges:
        if edge.has_point(point):
          return False
    
    # Interior
    intersection_points = []
    for edge in self.edges:
      try:
        inter = intersection(edge, half_line)
      except OnBoundary as e:
        intersection_points.append(e.intersection.pop())
      except GeometricException:
        pass
      else:
        intersection_points.append(inter)
    
    intersection_points_short = []
    for p1 in intersection_points:
      if not any(p1.is_eq(p2) for p2 in intersection_points_short):
        intersection_points_short.append(p1)

    if len(intersection_points_short)%2 == 0:
      return False
    else:
      return True

  def __repr__ (self):
    return "Polygon on points {}".format(self.points)
  
  def plot(self, figax=None, color='k', **kwargs):
    if not figax:
      figax = plt.subplots()
    fig, ax = figax
    for e in self.edges:
      e.plot(figax, color=color, **kwargs)

    return figax

  def triangularize(self):
    if self.lengh == 3:
      return set([Triangle(*[p for p in self.points])])
    elif len(set(self.points)) < self.lengh:
      for p in points:
        if points.count(p) > 1:
          break
      index_1, index_2 = self.points.index(p), self.points.index(p, index_1+1)
      points_1 = self.points[index_1:index_2]
      points_2 = self.points[index_2:] + self.points[:index_1]
      polygon_1, polygon_2 = Polygon(points_1), Polygon(points_2)
      return polygon_1.triangularize().union(polygon_2.triangularize())
    else:
      for i in range(self.lengh):
        p1, p2, p3 = [self.points[j % self.lengh] for j in range(i, i+3)]
        if self.has_point(Segment(p1, p3).mid_point):
          return set([Triangle(p1, p2, p3)]).union(self._remove_point(p2).triangularize())
      return set()
  
  def _remove_point(self, point):
    new_points = self.points.copy()
    new_points.remove(point)
    return Polygon(new_points)
