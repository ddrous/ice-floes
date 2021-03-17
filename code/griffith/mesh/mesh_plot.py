import sys, os
sys.path.append(os.path.abspath('..'))
import griffith.mesh as msh
import argparse

parser = argparse.ArgumentParser(description="Outputs *.msh file")
parser.add_argument('-m', '--mesh_name', help="Specify name of .msh file")

args = parser.parse_args()
m = msh.Mesh(args.mesh_name)
fig, ax = m.plot()
fig.savefig(args.mesh_name[:-4] + '.svg')
