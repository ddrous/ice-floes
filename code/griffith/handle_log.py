import argparse
import matplotlib.pyplot as plt
import re
from fabric import Connection
from io import BytesIO, StringIO
import subprocess
import logging

from griffith.mesh import Mesh, Fracture, Broken_Mesh_Linear_Tip
from griffith.geometry import Point, Segment

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
STATUS = 35 # between the two
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

LOG_NAMES = {DEBUG: "DEBUG", INFO: "INFO", WARNING: "WARNING", STATUS: "STATUS", ERROR: "ERROR", CRITICAL: "CRITICAL"}

def open_distant_file(name):
  c = Connection('tetta', user='balasoiu', gateway=Connection('ssh-ljk.imag.fr', user='balasoiu'))
  fd = BytesIO()
  c.get(name, fd)
  lines = fd.getvalue().decode('utf-8')
  return lines

def run(raw_args=None):
  parser = argparse.ArgumentParser(description='Outputs the solutions or errors in the log file')
  parser.add_argument('-pm', '--plot-mesh', action="store_true", help="Plot mesh")
  parser.add_argument('-sm', '--save-mesh', nargs='?', help="Saves mesh's plot, defaults to mesh.svg", const="mesh.svg")
  parser.add_argument('-log-level', '--log-level', help="Log level for display", default=STATUS)
  group = parser.add_mutually_exclusive_group()
  group.add_argument('-show', '--show', action="store_true", help='Show logs')
  group.add_argument('-err', '--error', action="store_true", help='Plots errors')
  group.add_argument('-sol', '--solution', action="store_true", help='Plots solution')
  group2 = parser.add_mutually_exclusive_group()
  group2.add_argument('-l', '--local', action="store_true", help='Local log')
  group2.add_argument('-d', '--distant', action="store_true", help='Distant log')
  
  args = parser.parse_args()
  assert any((args.local, args.distant)), "Specify location : local/distant"
  
  lines = get_log_file(args)
  mesh = get_mesh(args, lines[0][len('## Computations on '):])

  
  if args.show:
    lines = '\n'.join(filter_lines(args, lines))
    subprocess.run('less', input=lines, encoding='utf-8')
    print(lines)
    
  elif args.solution:
    result = next(iter(filter(lambda l: "Final" in l, lines)))
    if 'Fracture on points' in result:
      handle_segments(args, mesh, line_to_segments(result))
    else:
      print('No fracture')
  elif args.error:
    lines_errors = list(filter(lambda l: "ERROR" in l, lines))
    if len(lines_errors) > 1:
      print('Any preferences ?')
      for i, l in enumerate(lines_errors):
        print("{}. {}".format(i+1, l))
      choice = int(input("-> ")) - 1
      handle_segments(args, mesh, line_to_segments(lines_errors[choice]))

def get_log_file(args):
  if args.local:
    with open('griffith_solver.log', 'r') as log_file:
      lines = log_file.read().splitlines()
  else:
    assert args.distant
    lines = open_distant_file('griffith/griffith_solver.log').split('\n')
  return lines

def get_mesh(args, mesh_file):
  if args.local:
    mesh = Mesh(mesh_file)
  elif args.distant:
    mesh = Mesh(mesh_file=None, lines=open_distant_file('griffith/' + mesh_file))
  return mesh

def line_to_segments(line):
  line = re.sub(r'[a-zA-Z:(),-]*', '', line)
  coordinates = [float(p) for p in line.split()]
  points = [Point(x, y) for x, y in zip(*[iter(coordinates)]*2)]
  return [Segment(p1, p2) for p1, p2 in zip(points, points[1:])]

def filter_lines(args, lines):
  for level, name in LOG_NAMES.items():
    if level < args.log_level:
      lines = list(filter(lambda l: name not in l, lines))
  return lines

def handle_segments(args, mesh, segments):
  fracture = Fracture(segments, mesh)
  broken_mesh = Broken_Mesh_Linear_Tip(fracture, mesh)
  
  fig, ax = broken_mesh.plot()
  if args.plot_mesh:
    plt.show()
  if args.save_mesh:
    fig.savefig(args.save_mesh)

if __name__ == '__main__':
  run()  
