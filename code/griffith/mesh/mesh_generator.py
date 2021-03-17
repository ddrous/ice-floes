from subprocess import run
import argparse, warnings
import glob

parser = argparse.ArgumentParser(description="Operation on gmsh's *.geo files")
parser.add_argument('-g', '--geo_name', help="Specify name of .geo file, all if unspecified")

args = parser.parse_args()
if not args.geo_name:
  names = [f[:-4] for f in glob.glob("*.geo")]
else:
  names = [args.geo_name[:-4]]


for mesh_name in names:
  run('gmsh -2 {}.geo -o {}.msh -format msh2'.format(mesh_name, mesh_name), shell=True)
