import numpy as np
from scipy.sparse.linalg import LinearOperator, splu
import scipy.sparse as sparse


class Jacobian_Inverse(LinearOperator):
  def __init__(self, dEVdq, dVdp, cholesky=True, impact_index=None):
    self.shape = tuple(map(lambda x: 2*x, dEVdq.shape))
    assert dEVdq.dtype == dVdp.dtype
    self.dtype = dEVdq.dtype
    self.n = int(dEVdq.shape[0]/2)
    if cholesky:
      self.dEVdq_inverse = Cholesky_With_Swap_Impact(dEVdq, impact_index)
    else:
      self.dEVdq_inverse = SPLU_Inverse(dEVdq)
    self.dVdp = dVdp
  
  def _matvec(self, v):
    """
    Uses shur's formula for inverse of block matrix.
    """
    v1, v2 = v[0:2*self.n], v[2*self.n:4*self.n]
    return np.vstack([v2, self.dEVdq_inverse.dot(v1) - self.dEVdq_inverse.dot(self.dVdp.dot(v2))])


class Cholesky_With_Swap_Impact(LinearOperator):
  def __init__(self, dEVdq, impact_index):
    self.shape = dEVdq.shape
    self.dtype = dEVdq.dtype
    
    i = impact_index
    n, N = dEVdq.shape[0]/2, dEVdq.shape[0]
    self.P = sparse.eye(N).todok()
    self.P[i, i], self.P[0, 0], self.P[1, 1], self.P[i+n, i+n] = 0, 0, 0, 0
    self.P[i, 0], self.P[0, i], self.P[1, i+n], self.P[i + n, 1] = 1, 1, 1, 1
    self.P = self.P.tocsr()
    permuted = self.P.dot(dEVdq.dot(self.P))
    permuted_1, permuted_2 = permuted[0:2,:].tocsc(), permuted[2:,:].tocsc()
    self.A, self.B = permuted_1[:,0:2].toarray(), permuted_1[:,2:].toarray()
    self.C, self.D = permuted_2[:,0:2].toarray(), permuted_2[:,2:N]
    self.D_inv = Cholesky_Inverse(self.D)
    self.shur = np.linalg.inv(self.A - self.B.dot(self.D_inv.dot(self.C)))
  
  def _matvec(self, v):
    Pv = self.P.dot(v)
    v1, v2 = Pv[0:2], Pv[2:]
    return self.P.dot(np.hstack([self.shur.dot(v1) - self.shur.dot(self.B).dot(self.D_inv.dot(v2)),
                                -self.D_inv.dot(self.C.dot(self.shur)).dot(v1) + self.D_inv(v2) + self.D_inv.dot(self.C.dot(self.shur.dot(self.B.dot(self.D_inv.dot(v2)))))]))


class SPLU_Inverse(LinearOperator):
  def __init__(self, M):
    self.shape = M.shape
    self.dtype = M.dtype
    self.M_inv = splu(M.tocsc())
  
  def _matvec(self, v):
    return self.M_inv.solve(v)


class Cholesky_Inverse(LinearOperator):
  def __init__(self, M):
    self.shape = M.shape
    self.dtype = M.dtype
    from sksparse.cholmod import cholesky
    self.M_inv = cholesky(M.tocsc(), mode='simplicial')
    self.M_inv.solve_A(np.empty(self.shape[0]))  
  
  def _matvec(self, v):
    return self.M_inv.solve_A(v)
