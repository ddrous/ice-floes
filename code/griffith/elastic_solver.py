import argparse
import matplotlib.pyplot as plt
from griffith import solver, problem_data
from griffith.mesh import Mesh
import plotrc

def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Prints elastic energy (without fracture)')
  parser.add_argument('-m', '--mesh_file', default='mesh/square.msh', help="Name of the mesh file")
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-cd', '--constant-displacement', metavar=('TRACTION'), type=float, help="Constant displacement on Oy axis")
  group.add_argument('-ld', '--linear-displacement', metavar=('TRACTION', 'YMIN', 'CMIN', 'YMAX', 'CMAX'), type=float, nargs=5, help="Linear displacement on Oy axis")
  group.add_argument('-pld', '--picewise-linear-displacement', type=float, metavar=('TRACTION', 'YMIN', 'YMAX'), nargs=3, help="Linear displacement on Oy axis")
  group.add_argument('-rd', '--rotation-displacement', metavar=('ANGLE', 'LEFT-P.X', 'LEFT-P.Y', 'RIGHT-P.X', 'RIGHT-P.Y'), type=float, nargs=5, help="Rotation displacement on Oy axis")
  parser.add_argument('-pd', '--plot-displacement', action="store_true", help="Plot displacement")
  parser.add_argument('-pe', '--plot-energy', action="store_true", help="Plot energy")
  parser.add_argument('-pm', '--plot-mesh', action="store_true", help="Plot mesh")
  parser.add_argument('-se', '--save-energy', nargs='?', const="energy.svg", help="Save energy")
  parser.add_argument('-sm', '--save-mesh', nargs='?', help="Saves mesh's plot, defaults to mesh.svg", const="mesh.svg")
  args = parser.parse_args(raw_args)
  
  ###############
  # Physical Data
  ###############
  mesh = Mesh(mesh_file=args.mesh_file)
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
  physical_data = problem_data.Physical_Data(stiffness_tensor=T, toughness_field=None, boundary_displacement=boundary_displacement)
  
  ########
  # Solver
  ########
  classical_solution = solver.Classical_Solution(mesh=mesh, physical_data=physical_data)
  print(classical_solution.energy)


  if args.plot_mesh:
    fig, ax = classical_solution.mesh.plot()
    plt.show()
  if args.save_mesh:
    fig, ax = classical_solution.mesh.plot()
    fig.savefig(args.save_mesh, **plotrc.savefig)
  if args.plot_displacement:
    fig, ax = classical_solution.plot_displacement()
    plt.show()
  if args.plot_energy or args.save_energy:
    fig, ax = classical_solution.plot_energy()
    if args.plot_energy:
      plt.show()
    else:
      fig.savefig(args.save_energy, **plotrc.savefig)
    
if __name__ == '__main__':
  run()
