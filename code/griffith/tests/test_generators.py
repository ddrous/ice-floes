import pytest
import admissible_fractures

def test_fracture_generator_bp_iterator():
  admissible_fractures.run(['BPI', '-bs', '10'])

def test_fracture_generator_ip_iterator():
  admissible_fractures.run(['IPI', '-is', '40'])

def test_fracture_generator_bs_iterator():
  admissible_fractures.run(['BSI', '-bs', '50', '-ls', '50'])

def test_fracture_generator_bs_iterator_fixed_bp():
  admissible_fractures.run(['BSI', '-bp', '50', '0', '-ls', '10'])

def test_fracture_generator_bs_iterator_fixed_bp_2():
  admissible_fractures.run(['BSI', '-bp', '50', '0', '-ls', '1000'])

def test_fracture_generator_is_iterator():
  admissible_fractures.run(['ISI', '-is', '30', '-ls', '30', '-as', '1', '-isa', '2'])

def test_fracture_generator_polyline_boundary():
  admissible_fractures.run(['EFI', '-p', '50', '0', '45', '60', '-ls', '40'])

def test_fracture_generator_polyline_interior():
  admissible_fractures.run(['EFI', '-p', '50', '30', '45', '60', '-ls', '70'])
