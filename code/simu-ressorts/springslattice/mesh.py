import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def Poisson_PP(rate, a, b=None):
  """
  Returns a draw of poisson point process,
  with points (x,y) in the following rectangle:
  0 < x < a
  0 < y < b
  If b isn't specified, we take b=a.
  """
  if b is None:
    b = a
  
  N = np.random.poisson(rate*a*b) # because mean = rate * volume
  X = np.random.uniform(0, a, N)
  Y = np.random.uniform(0, b, N)
  PP = list(zip(X,Y))
  return X,Y,PP

def euclidian(a, b):
  return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def admissible_poisson_draw(rate):
  X, Y, PP = Poisson_PP(rate, 1)
  while len(X) < 4:
    X, Y, PP = Poisson_PP(rate, 1)
  return X, Y, PP

def scale_triangulation(triangulation):
  dmin, dmax = 10, 0
  p = triangulation.points
  for t in triangulation.simplices:
    i1, i2, i3 = t
    d1, d2, d3 = euclidian(p[i1], p[i2]), euclidian(p[i2], p[i3]), euclidian(p[i1], p[i3])
    for d in d1, d2, d3:
      if d < dmin:
        dmin = d
      if d > dmax:
        dmax = d
  return dmin/dmax

def regularity_triangulation(triangulation):
  r = 100
  p = triangulation.points
  for t in triangulation.simplices:
    i1, i2, i3 = t
    d1, d2, d3 = euclidian(p[i1], p[i2]), euclidian(p[i2], p[i3]), euclidian(p[i1], p[i3])
    dmin = min([d1, d2, d3])
    dmax = max([d1, d2, d3])
    rloc = dmin/dmax
    if r > rloc:
      r = rloc
  return r

def DelaunayTriangulation(rate):
  X, Y, PP = admissible_poisson_draw(rate)
  return Delaunay(PP)


class Mesh:
  def __init__(self, rate, regularity):
    self.nodes = []
    self.triangles = []
    self.edges = []
    
    self._tri = DelaunayTriangulation(rate)
    if regularity != (0, 1):
      while True:
        self._tri = DelaunayTriangulation(rate)
        self._regularity = regularity_triangulation(self._tri)
        if self._regularity > regularity[0] and self._regularity < regularity[1]:
          break

    for i, node in enumerate(self._tri.points):
      self.nodes.append(Node(node[0], node[1], i))

    for i, t in enumerate(self._tri.simplices):
      t_nodes = [self.nodes[t[0]], self.nodes[t[1]], self.nodes[t[2]]]
      t_current = Triangle(t_nodes, i)
      self.triangles.append(t_current)

      for node in t_nodes:
        node.of_triangles.add(t_current)
        node.add_neighbors(set(t_nodes) - set([node]))
        
    for n1 in self.nodes:
      for n2 in n1.neighbors:
        if n2.id_number < n1.id_number:
          self.edges.append(Line(n1, n2, len(self.edges)))
        
    self.boundary_mesh = Boundary_Mesh(self.edges)

  def d_eq(self, node_1, node_2):
    # Distance between two nodes at equilibrium
    return np.sqrt((node_1.x-node_2.x)**2+(node_1.y-node_2.y)**2)

  def save_hdf5(self, data_group=None):
    import h5py
    if not data_group:
      with h5py.File('mesh.hfd5', 'w') as data_file:
        self._save_hdf5(data_file)
    else:
      self._save_hdf5(data_group)

  @property
  def regularity(self):
    try:
      r = self._regularity
    except AttributeError:
      r = regularity_triangulation(self._tri)
    return r
  
  def _save_hdf5(self, data_group):
    data_nodes = data_group.create_dataset('Nodes', (len(self.nodes),2)) 
    data_triangles = data_group.create_dataset('Triangles', (len(self.triangles),3)) 
    data_nodes[:,:], data_triangles[:,:] = self.get_data_mesh()

  def get_data_mesh(self):
    nodes_to_list = list(map(Node.to_list, self.nodes))
    triangles_to_list = list(map(Triangle.to_list, self.triangles))
    return nodes_to_list, triangles_to_list
  
  def plot(self, figax=None, save_file=None):
    if figax:
      fig, ax = figax
    else:
      figax = plt.subplots()
      fig, ax = figax
      ax.axis([0, 1, 0, 1])

    for p in self.nodes:
      ax.plot(p.x, p.y, 'k.')
    
    for e in self.edges:
      color='r' if e in self.boundary_mesh.edges else 'b'
      p1, p2 = e.points
      x = [p1.x, p2.x]
      y = [p1.y, p2.y]
      ax.plot(x, y, color)
    
    return figax


class Boundary_Mesh:
  def __init__(self, edges):
    self.edges = set([e for e in edges if e.on_boundary])
    self.nodes = set([p for e in self.edges for p in e.points])
  
  def neighbors(self, p):
    assert p in self.nodes
    return (q for q in p.neighbors if q in self.nodes)
    

class Node:
  def to_list(self):
    return [self.x, self.y]
    
  def to_array(self):
    return np.array([self.x, self.y])

  def __init__(self, x, y, id_number):
    self.x = x
    self.y = y
    self.id_number = id_number
    self.of_triangles = set()
    self.neighbors = set()
    
  def add_neighbors(self, nodes):
    self.neighbors.update(nodes)
    
  def is_neighbor(self, node):
    if node in self.neighbors:
      return True
    else:
      return False

    
class Triangle:
  @staticmethod
  def to_list(t):
    return [n.id_number for n in t.nodes]

  def __init__(self, nodes, id_number):
    assert (len(nodes) == 3)
    self.nodes = set(nodes)
    self.id_number = id_number


class Line:
  def __init__(self, n1, n2, id_number):
    self.points = set((n1, n2))
    self.mid = 0.5*np.array((n1.x + n2.x, n1.y + n2.y))
    if len(n1.of_triangles & n2.of_triangles) == 1:
      self.on_boundary = True
    else:
      self.on_boundary = False
