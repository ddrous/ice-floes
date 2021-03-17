import time
import argparse
import matplotlib.pyplot as plt
from math import pi

import plotrc
from griffith import solver, log, problem_data
from griffith.mesh import Mesh
from griffith.geometry import Point, Segment

def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Simulates the fracturation of an elastic material')
  parser.add_argument('-f', '--fracture', type=str, help="Initial fracture in material")
  parser.add_argument('-np', '--number-processes', type=int, help="Number processes")
  parser.add_argument('-m', '--mesh-file', default='mesh/square.msh', help="Name of the mesh file")
  parser.add_argument('-k', '--toughness', type=float, default=1, help="Toughness")
  parser.add_argument('-ci', '--circle-inclusion', type=float, help="Circle inclusion toughness")
  parser.add_argument('-as', '--angular-step', type=float, default=pi/4, help="Angle step for fracture discretisation")
  parser.add_argument('-ls', '--lengh-step', type=float, default=10, help="Lengh step for fracture discretisation")
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-bs', '--boundary-step', type=float, help="Boundary step for fracture discretisation")
  parser.add_argument('-is', '--interior-step', type=float, help="Interior step for fracture discretisation") 
  parser.add_argument('-isa', '--interior-start-angle', type=float, help="Interior start angle for fracture discretisation")
  parser.add_argument('-ifi', '--interior-fast-init', action='store_true', help="Interior fast initialisation")
  parser.add_argument('-ifs', '--interior-fast-step', action='store_true', help="Interior fast step")
  group.add_argument('-bp', '--boundary-point', type=float, nargs=2, help="Boundary point for fracture initiation")
  group2 = parser.add_mutually_exclusive_group()
  group2.add_argument('-cd', '--constant-displacement', metavar=('TRACTION'), type=float, help="Constant displacement on Oy axis")
  group2.add_argument('-ld', '--linear-displacement', metavar=('TRACTION', 'YMIN', 'CMIN', 'YMAX', 'CMAX'), type=float, nargs=5, help="Linear displacement on Oy axis")
  group2.add_argument('-pld', '--picewise-linear-displacement', type=float, metavar=('TRACTION', 'YMIN', 'YMAX'), nargs=3, help="Linear displacement on Oy axis")
  group2.add_argument('-rd', '--rotation-displacement', metavar=('ANGLE', 'LEFT-P.X', 'LEFT-P.Y', 'RIGHT-P.X', 'RIGHT-P.Y'), type=float, nargs=5, help="Rotation displacement on Oy axis")
  group3 = parser.add_mutually_exclusive_group()
  group3.add_argument('-st', '--smart-time', action="store_true", help='Smart-Time evolution algorithm')
  group3.add_argument('-fts', '--fixed-time-step', type=float, metavar=('TIME-STEP', 'TIME_MIN'), nargs=2, help='Fixed time-step discretization of the time/loading')
  parser.add_argument('-pm', '--plot-mesh', action="store_true", help="Plot mesh")
  parser.add_argument('-pd', '--plot-displacement', action="store_true", help="Plot displacement field")
  parser.add_argument('-sm', '--save-mesh', nargs='?', help="Saves mesh's plot, defaults to mesh.svg", const="mesh.svg")
  parser.add_argument('-sd', '--save-displacement', nargs='?', help="Save displacement field, defaults to displacement.svg", const="displacement.svg")
  args = parser.parse_args(raw_args)
  
  #########
  # Logging
  #########
  logger = log.Log('griffith_solver.log', level=log.INFO, console_output=True)
  logger.log_description(mesh_file=args.mesh_file, args=args)
  log_queue = logger._log_queue
  
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
    toughness_field = problem_data.Smooth_Japan_Toughness(args.toughness, args.circle_inclusion, center=(50, 50), radius=10)
  else:
    toughness_field = problem_data.Constant_Toughness(args.toughness)
  physical_data = problem_data.Physical_Data(T, toughness_field, boundary_displacement, initial_fracture=args.fracture)
  
  #####################
  # Discretization Data
  #####################
  mesh = Mesh(args.mesh_file)
  if args.fixed_time_step:
    time_step, time_min = args.fixed_time_step
    time_discretization = problem_data.Regular_Time_Discretization(time_step=time_step, time_min=time_min)
  else:
    time_discretization = None
    if not args.smart_time:
      parser.error('No time discretization')
  
  fracture_discretization = problem_data.Fracture_Discretization(args.angular_step, args.lengh_step, boundary_point=args.boundary_point, boundary_step=args.boundary_step,
                                                                 interior_step=args.interior_step, interior_fast_init=args.interior_fast_init, interior_fast_step=args.interior_fast_step, interior_start_angle=args.interior_start_angle)
  discretization_data =  problem_data.Discretization_Data(mesh, time_discretization, fracture_discretization, tip_enrichement=False)
  
  ########
  # Solver
  ########
  if args.smart_time:
    solution = solver.smart_time_solver(discretization_data, physical_data, log_queue, args.number_processes)
  else:
    solution = solver.solver_with_time_discretization(discretization_data, physical_data, log_queue, args.number_processes)
  
  time.sleep(0.1)
  
  log_queue.put(('INTRO', 'Final Fracture : {}'.format(solution.fracture)))
  if args.plot_displacement:
    fig, ax = solution.plot_displacement()
    plt.show()
  if args.plot_mesh:
    fig, ax = solution.plot()
    plt.show()
  if args.save_displacement:
    fig, ax = solution.plot_displacement()
    fig.savefig(args.save_displacement, **plotrc.savefig)
  if args.save_mesh:
    fig, ax = solution.plot()
    fig.savefig(args.save_mesh, **plotrc.savefig)
  logger.exit()

if __name__ == '__main__':
  run()  
