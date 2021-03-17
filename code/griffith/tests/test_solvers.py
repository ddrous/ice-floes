import pytest
import elastic_solver, broken_mesh_solver, griffith_solver

linear_boundary_conditions = [['-cd', '1'], ['-ld', '1', '0', '1', '100', '1'], ['-pld', '1', '0', '50']]
boundary_conditions = linear_boundary_conditions + [['-rd', '0.1', '0', '0', '100', '0']]
constant_displacement = ['-cd', '1']
linear_displacement = ['-ld', '1', '0', '1', '100', '1']
picewise_linear_displacement = ['-pld', '1', '0', '50']
rotation_displacement = ['-rd', '0.1', '0', '0', '100', '0']
fracture = ['-f', '50 0 50 40']
fracture_discretization = ['-as', '1', '-ls', '60', '-bp', '50', '0']
smart_time = ['-st']
fixed_time_step = ['-fts', '0.5', '0.2']

def test_elastic_solver_constant_displacement():
  elastic_solver.run(constant_displacement)

def test_elastic_solver_linear_displacement():
  elastic_solver.run(linear_displacement)

def test_elastic_solver_picewise_linear_displacement():
  elastic_solver.run(picewise_linear_displacement)

def test_elastic_solver_rotation_displacement():
  elastic_solver.run(rotation_displacement)

def test_broken_mesh_solver_constant_displacement():
  broken_mesh_solver.run(constant_displacement + fracture)

def test_broken_mesh_solver_linear_displacement():
  broken_mesh_solver.run(linear_displacement + fracture)

def test_broken_mesh_solver_picewise_linear_displacement():
  broken_mesh_solver.run(picewise_linear_displacement + fracture)

def test_broken_mesh_solver_rotation_displacement():
  broken_mesh_solver.run(rotation_displacement + fracture)

def test_griffith_solver_smart_time_constant_displacement():
  griffith_solver.run(smart_time + fracture_discretization + constant_displacement)

def test_griffith_solver_smart_time_linear_displacement():
  griffith_solver.run(smart_time + fracture_discretization + linear_displacement)

def test_griffith_solver_smart_time_picewise_linear_displacement():
  griffith_solver.run(smart_time + fracture_discretization + picewise_linear_displacement)

def test_griffith_solver_fixed_time_step_constant_displacement():
  griffith_solver.run(fixed_time_step + fracture_discretization + constant_displacement)

@pytest.mark.filterwarnings("ignore:.*singular.*")
def test_griffith_solver_fixed_time_step_linear_displacement():
  griffith_solver.run(fixed_time_step + fracture_discretization + linear_displacement)

def test_griffith_solver_fixed_time_step_picewise_linear_displacement():
  griffith_solver.run(fixed_time_step + fracture_discretization + picewise_linear_displacement)

def test_griffith_solver_fixed_time_step_rotation_displacement():
  griffith_solver.run(fixed_time_step + fracture_discretization + rotation_displacement)

def test_griffith_solver_interior():
  griffith_solver.run(fixed_time_step + ['-is', '40', '-ls', '60', '-isa', '2', '-ifi'] + constant_displacement)

