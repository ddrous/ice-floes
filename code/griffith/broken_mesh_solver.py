import argparse
import matplotlib.pyplot as plt
import scipy

from griffith import solver, problem_data 
from griffith.mesh import Mesh, Fracture, Broken_Mesh_Linear_Tip
from griffith.geometry import Point, Segment
import plotrc


def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Simulates fracturation with fixed fracture')
  parser.add_argument('-f', '--fracture', type=str, help="Entering fracture points")
  parser.add_argument('-k', '--toughness', type=float, default=1, help="Toughness")
  parser.add_argument('-ci', '--circle-inclusion', type=float, help="Circle inclusion toughness")
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-cd', '--constant-displacement', metavar=('TRACTION'), type=float, help="Constant displacement on Oy axis")
  group.add_argument('-ld', '--linear-displacement', metavar=('TRACTION', 'YMIN', 'CMIN', 'YMAX', 'CMAX'), type=float, nargs=5, help="Linear displacement on Oy axis")
  group.add_argument('-pld', '--picewise-linear-displacement', type=float, metavar=('TRACTION', 'YMIN', 'YMAX'), nargs=3, help="Linear displacement on Oy axis")
  group.add_argument('-rd', '--rotation-displacement', metavar=('ANGLE', 'LEFT-P.X', 'LEFT-P.Y', 'RIGHT-P.X', 'RIGHT-P.Y'), type=float, nargs=5, help="Rotation displacement on Oy axis")
  parser.add_argument('-m', '--mesh-file', default='mesh/square.msh', help="Name of the mesh file")
  parser.add_argument('-pd', '--plot-displacement', action="store_true", help="Plot displacement")
  parser.add_argument('-pm', '--plot-mesh', action="store_true", help="Plot mesh")
  parser.add_argument('-pe', '--plot-energy', action="store_true", help="Plot energy")
  parser.add_argument('-se', '--save-energy', nargs='?', const="energy.svg", help="Save energy")
  parser.add_argument('-sm', '--save-mesh', nargs='?', help="Saves mesh's plot, defaults to mesh.svg", const="mesh.svg")
  args = parser.parse_args(raw_args)
  
  ###############
  # Physical Data
  ###############
  T = problem_data.lame_tensor_ice
  if args.constant_displacement:
    boundary_displacement = problem_data.Constant_Displacement_On_Y(traction_coefficient=args.constant_displacement)
  elif args.picewise_linear_displacement:
    traction_coefficient, y_min, y_max = args.picewise_linear_displacement
    boundary_displacement = problem_data.Picewise_Linear_Displacement_On_Y(traction_coefficient=traction_coefficient, y_min=y_min, y_max=y_max)
  elif args.linear_displacement:
    traction_coefficient, y_min, c_min, y_max, c_max = args.linear_displacement
    boundary_displacement = problem_data.Linear_Displacement_On_Y(traction_coefficient=traction_coefficient, y_min=y_min, c_min=c_min, y_max=y_max, c_max=c_max)
  elif args.rotation_displacement:
    angle, p1x, p1y, p2x, p2y = args.rotation_displacement
    boundary_displacement = problem_data.Rotation_Displacement_On_Y(angle=angle, point_left=(p1x, p1y), point_right=(p2x, p2y))
  else:
    parser.error('No displacement on dirichlet boundary')
  if args.circle_inclusion:
    toughness_field = problem_data.Japan_Toughness(args.toughness, args.circle_inclusion, center=(50, 50), radius=10)
  else:
    toughness_field = problem_data.Constant_Toughness(args.toughness)
  physical_data = problem_data.Physical_Data(stiffness_tensor=T, toughness_field=toughness_field, boundary_displacement=boundary_displacement)
  
  ########
  # Solver
  ########
  mesh = Mesh(mesh_file=args.mesh_file)
  classical_solution = solver.Classical_Solution(mesh=mesh, physical_data=physical_data)
  fracture = Fracture.from_str(args.fracture, mesh)
  try:
    fractured_solution = solver.Imposed_Fracture_Solution.from_classical_solution(classical_solution, physical_data, fracture, tip_enrichement=False)
    #fractured_solution = solver.Imposed_Fracture_Solution(mesh, physical_data, fracture, tip_enrichement=args.tip_enrichement)
  except scipy.linalg.LinAlgWarning:
    fractured_solution = solver.Infinite_Energy()
    if args.tip_enrichement:
      fractured_solution.mesh = Broken_Mesh_Nonlinear_Tip(fracture, mesh)
    else:
      fractured_solution.mesh = Broken_Mesh_Linear_Tip(fracture, mesh)
  
  if args.plot_mesh:
    fig, ax = fractured_solution.mesh.plot()
    plt.show()
  if args.save_mesh:
    fig, ax = fractured_solution.mesh.plot()
    fig.savefig(args.save_mesh, **plotrc.savefig)
  if args.plot_displacement:
    fig, ax = classical_solution.plot_displacement()
    fig, ax = fractured_solution.plot_displacement()
    plt.show()
  if args.plot_energy or args.save_energy:
    fig, ax = classical_solution.plot_energy()
    fig, ax = fractured_solution.plot_energy()
    if args.plot_energy:
      plt.show()
    else:
      fig.savefig(args.save_energy, **plotrc.savefig)

  print('Energy without fracture: {}'.format(classical_solution.energy))
  print('Energy with fracture: {}'.format(fractured_solution.energy))
  print('  |_ Elastic energy: {}  Fracture energy: {}'.format(fractured_solution.elastic_energy, fractured_solution.fracture_energy))

if __name__ == '__main__':
  run()
