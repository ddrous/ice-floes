import pytest
from griffith.geometry import Point, Segment
from griffith.mesh import Mesh, Fracture, Broken_Mesh_Linear_Tip

def generate_broken_mesh(points):
  segments = [Segment(p1, p2) for p1, p2 in zip(points, points[1:])]
  mesh = Mesh('tests/square.msh')
  fracture = Fracture(segments, mesh)
  broken_mesh = Broken_Mesh_Linear_Tip(fracture, mesh)

def test_fracture_linear_tip_case_1_1():
  points = [Point(50, 0), Point(50, 30)]
  generate_broken_mesh(points)

def test_fracture_linear_tip_case_1_2():
  points = [Point(50, 0), Point(50, 51)]
  generate_broken_mesh(points)

@pytest.mark.skip()
def test_fracture_linear_tip_case_2():
  points = [Point(47.555267117154806, 0.0), Point(49.77610803459987, 45.420931935482855)]
  generate_broken_mesh(points)

def test_fracture_linear_tip_case_3():
  points = [Point(50, 0), Point(50, 31.25525)]
  generate_broken_mesh(points)

def test_fracture_two_end_points():
  points = [Point(50, 30), Point(50, 60)]
  generate_broken_mesh(points)
