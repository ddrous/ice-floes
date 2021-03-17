import sys
import argparse
sys.path.insert(0, '..')     

from griffith import mesh as msh
from griffith import geometry as geo
parser = argparse.ArgumentParser(description='Plots simulation results')
parser.add_argument('-f', '--folder', nargs='?', type=str, help="Folder to plot")
args = parser.parse_args()

args.folder.replace('/', '')

with open(args.folder + '/' + 'griffith_solver.log' , 'r') as txt:
  mesh_file = txt.readline().replace('## Computations on ', '').replace('\n', '').replace('mesh/', '')
  parameters = txt.readline()
  mesh = msh.Mesh(args.folder + '/' + mesh_file)
  i1 = parameters.find('circle_inclusion') + len('circle_inclusion=')
  if parameters[i1:i1+4] == 'None':
    circle = None 
  else:
    circle = geo.Circle(geo.Point(50, 50), 10)

  i = 0
  for line in txt.readlines():
    if 'Time' in line:
      k = line.find(':')
      fracture = msh.Fracture.from_str(line[k+1:], mesh)
      f, a = mesh.plot()
      fracture.plot((f, a))
      if circle:
        circle.plot((f, a))
      f.savefig(args.folder + '/' + args.folder + '-{}'.format(i) + '.pdf', bbox_inches='tight', pad_inches=0)
      i += 1
