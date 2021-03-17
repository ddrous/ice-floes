import pytest
from springslattice import solver

def test_jacobian():
  n, m, im, k, v, iv = 15, 10, 10, 10, 100, 0.1

  for n in [10, 10, 10]:
    sn = solver.SpringNetwork(n, m, k, v, iv, im)
    for ep in [1, 1/10, 1/100, 1/1000]:
      sn.test_explicit_jacobian()

    
