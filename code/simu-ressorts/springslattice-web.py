import matplotlib as mpl
mpl.use('Agg')

from web import app
from springslattice import solver

class Backend:
  def __init__(self):
    self.sn = solver.SpringNetwork(15, 10, 10, 0, 0.1, 10)
    self.subpackage_path = None
    self.sol = None
    self.final_time = 10

  @property
  def number_particles(self):
    return self.sn.n
  
  @property
  def data(self):
    return {'pi': self.sn.intensity,
            'm': self.sn.total_mass,
            'k': self.sn.k,
            'v': self.sn.v,
            'iv': self.sn.impact_velocity,
            'im': self.sn.impact_mass,
            'tf': self.final_time}
  
  @property
  def number_eigenva(self):
    try:
      nbr = len(self.sn.eigva)
    except AttributeError:
      nbr = 0
    return nbr 
  
  def reset_constants(self, mass, stiffness, viscosity, impact_velocity, impact_mass):
    self.sn.reset_constants(mass, stiffness, viscosity, impact_velocity, impact_mass)
    self.sol = None

  def redraw(self, intensity):
    self.sn = solver.SpringNetwork(intensity, self.sn.total_mass, self.sn.k, self.sn.v, self.sn.impact_velocity, self.sn.impact_mass)
    self.sol = None

  def compute_sol(self, final_time):
    if not self.sol:
      #self.sol = self.sn.compute_sol_lin((0, final_time))
      self.sol = self.sn.compute_sol((0, final_time))
      self.final_time = final_time
    
  def compute_df(self, final_time, path):
    self.compute_sol(final_time)
    self.sn.plot_solution(self.sol, path)
  
  def plot_eigenva(self, path):
    self.sn.plot_eig(path)
    
  def plot_mesh(self, path):
    self.sn.plot_mesh(path)

app.backend = Backend()
app.run(port=8080, debug=True)
