from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import warn_extraneous, validate_max_step, validate_tol
from scipy.optimize import fsolve
import numpy as np

class SIE(OdeSolver):
  def __init__(self, fun, t0, y0, t_bound, max_step=np.inf, rtol=1e-3, atol=1e-6, vectorized=False, h_abs=1, **extraneous):
    warn_extraneous(extraneous)
    super(SIE, self).__init__(fun, t0, y0, t_bound, vectorized, support_complex=False)
    self.max_step = validate_max_step(max_step)
    self.rtol, self.atol = validate_tol(rtol, atol, self.n)
    self.h_abs = h_abs
    
  def _step_impl(self):
    t = self.t
    y = self.y
    fun = self.fun
    h_abs = self.h_abs
    
    self.y_old = y
    self.t_old = t

    t_new = t + h_abs
    y_new = verlet_splitted(fun, t, y, h_abs)
    self.t = t_new
    self.y = y_new

    return True, None

  def _dense_output_impl():
    pass


def symplectic_euler(fun, t, y, h, jac=None):
  n = int(len(y)/2)
  p = y[0:n]
  q = y[n:2*n]
    
  def f(x):
    return x - h*fun(t, np.hstack((x, q)))[0:n]
  
  p_new = fsolve(f, p, fprime = jac) 
  q_new = q + h*fun(t, np.hstack((p_new, q)))[n:2*n]
  y_new = np.hstack((p_new, q_new))
  return y_new


def verlet_splitted(fun, t, y, h):
  n = int(len(y)/2)
  p = y[0:n]
  q = y[n:2*n]
  r = np.zeros((n))

  p12 = p + 0.5*h*fun(t, np.hstack((p, q)))[0:n]
  q_new = q + h*fun(t, np.hstack((p12, q)))[n:2*n]
  p_new = p12 + 0.5*h*fun(t, np.hstack((p, q_new)))[0:n]
  
  return np.hstack((p_new, q_new))
