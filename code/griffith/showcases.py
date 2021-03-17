#!/usr/bin/env python3
import sys, os
from threading import Thread
from queue import Queue, Empty
from subprocess import run, CalledProcessError
import signal


cl1 = 'python3 griffith_solver.py -sm traction1.pdf -bp 50 0 -ls 1000 -cd 1 -st' 
cl2 = 'python3 griffith_solver.py -sm traction2.pdf -bp 50 0 -ls 1000 -cd 1000 -st'
cl3 = 'python3 griffith_solver.py -m mesh/butterfly.msh -sm boundary1.pdf -ls 1000 -as 1 -cd 1000 -bs 70 -st'
cl4 = 'python3 griffith_solver.py -m mesh/butterfly.msh -sm boundary2.pdf -ls 1000 -as 1 -cd 1000 -bs 5 -np 4 -st'
cl5 = 'python3 griffith_solver.py -m mesh/butterfly.msh -sm angular1.pdf -ls 1000 -bp 49.973520189271696 28.307566540355406 -cd 1000 -as 1 -st'
cl6 = 'python3 griffith_solver.py -m mesh/butterfly.msh -sm angular2.pdf -ls 1000 -bp 49.973520189271696 28.307566540355406 -cd 1000 -as 0.1 -np 4 -st'
cl7 = 'python3 breaking_point.py -m mesh/rectangle.msh -bp 4.8 0 -ls 10 -cd 10'
cl8 = 'python3 breaking_point.py -m mesh/square.msh -bp 50 0 -ls 10 -cd 10'
cl9 = 'python3 griffith_solver.py -m mesh/square.msh -sm cl9.png -bp 50 0 -k 12 -ld 30 0 2 100 0.5 -fts 0.1 0.1 -ls 10 -as 1 -np 4 -ci 500' 
cl10 = 'python3 griffith_solver.py -m mesh/square-tight.msh -sm cl10.png -bp 50 0 -k 12 -ld 30 0 2 100 0.5 -fts 0.1 0.1 -ls 10 -as 1 -np 4 -ci 500' 

def call(cmd):
  try:
    run(cmd, shell=True, check=True)
  except CalledProcessError:
    print('#WARNING: some fracture simulations went wrong, check the log')


#############
# Simulations
#############
def traction():
  print("""
  *********************
  Influence of Traction
  *********************
  """)
  
  print("""
  -> Case 1:
  Fracturation simulation of a rectangular ice-floe with:
    - only traversant fractures
    - initiation on fixed boundary point
    - small traction
  Command line: {}
  Result: cl1.png file
  """.format(cl1))
  call(cl1)
  
  print("""
  -> Case 2:
  Fracturation simulation of a rectangular ice-floe with
    - only traversant fractures
    - initiation on fixed boundary point
    - with huge traction
  Command line: {}
  Result: cl2.png file
  """.format(cl2))
  call(cl2)

def boundary_precision():
  print("""
  **************************
  Influence of boundary step
  **************************
  """)
  
  print("""
  -> Case 3:
  Fracturation simulation of a butterfly-shaped ice floe with
    - only traversant fractures
    - large boundary step
    - huge traction
  Command line: {}
  Result: cl3.png file
  """.format(cl3))
  call(cl3)
  
  print("""
  -> Case 4:
  Fracturation simulation of a butterfly-shaped ice floe with
    - only traversant fractures
    - large boundary step
    - huge traction
  Command line: {}
  Result: cl4.png file
  """.format(cl4))
  call(cl4)

def angle_precision():
  print("""
  *************************
  Influence of angular step
  *************************
  """)
  
  print("""
  -> Case 5:
  Fracturation simulation of a butterfly-shaped ice floe with
    - only traversant fractures
    - fixed boundary point
    - large angular step
    - huge traction
  Command line: {}
  Result: cl5.png file
  """.format(cl5))
  call(cl5)
  
  print("""
  -> Case 6:
  Fracturation simulation of a butterfly-shaped ice floe with
    - only traversant fractures
    - fixed boundary point
    - large angular step
    - huge traction
  Command line: {}
  Result: cl6.png file
  """.format(cl6))
  call(cl6)

def geometry():
  print("""
  *********************
  Influence of geometry
  *********************
  """)
  
  print("""
  -> Case 7:
  Find instable point of rectangular mesh (100x10) with traction on Dirichlet-side
  DND
  D D
  D D
  D D
  D D
  D D
  D D
  D D
  D D
  DND
  Command line: {}
  """.format(cl7))
  call(cl7)
  
  print("""
  -> Case 8:
  Find instable point of square mesh (100x100) with traction on Dirichlet-side
  DNNNNNNNND
  D        D
  D        D
  D        D
  D        D
  D        D
  D        D
  D        D
  D        D
  DNNNNNNNND
  Command line: {}
  """.format(cl8))
  call(cl8)

def inclusion():
  print("""
  *********************
  Influence of geometry
  *********************
  """)

  print("""
  -> Case 9:
  Fracturation simulation of a squared ice floe with circular inclusion
  Command line: {}
  Result: cl9.png file
  """.format(cl9))
  call(cl9)
  
  print("""
  -> Case 6:
  Fracturation simulation of a squared ice floe with circular inclusion on tight mesh
  Command line: {}
  Result: cl10.png file
  """.format(cl10))
  call(cl10)


######
# Menu
######
print("""Some test cases for the griffith solver. We have :
  1. Influence of traction
  2. Influence of boundary precision
  3. Influence of angle precision
  4. Influence of geometry #NONWORKING
  5. Influence of circular inclusion""")

choice = input('Enter a choice: ')
choice = int(choice)
if choice == 1:
  traction()
elif choice == 2:
  boundary_precision()
elif choice == 3:
  angle_precision()
elif choice == 4:
  geometry()
elif choice == 5:
  inclusion()

