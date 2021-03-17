# cython: profile=True

import numpy as np
from itertools import chain
import scipy.sparse as sparse

cimport numpy as np
from libc.math cimport sqrt
cimport cython

from .inverse import Jacobian_Inverse

nfloat = np.float64
ctypedef np.float64_t nfloat_t

cdef nfloat_t euclidian(nfloat_t ax, nfloat_t ay, nfloat_t bx, nfloat_t by):
  return np.sqrt((ax-bx)**2 + (ay-by)**2)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def jacobian_loop(n, nfloat_t k, nfloat_t[:] m, nfloat_t v, nfloat_t[:] L0, row, col, nfloat_t[:] qx, nfloat_t[:] qy, nfloat_t[:] dqx, nfloat_t[:] dqy, reduced, reduced_index, fix_indexes, impact_index, with_inverse, with_cholesky):
  cdef int k1 = 0
  cdef int k2 = 0
  cdef int i, j
  cdef int tot_len
  cdef int row_len
  
  row_len = len(row)
  if reduced:
    row2, col2 = list(zip(*filter(lambda x: x[0] not in fix_indexes and x[1] not in fix_indexes, zip(row, col))))
    row2_len = len(row2)
    tot_len = row_len + row2_len
  else:
    tot_len = 2*row_len
  
  cdef np.ndarray[nfloat_t] dEVxdqx =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dEVxdqy =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dEVydqx =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dEVydqy =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dVxdpx =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dVxdpy =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dVydpx =  np.zeros(tot_len, dtype=nfloat)
  cdef np.ndarray[nfloat_t] dVydpy =  np.zeros(tot_len, dtype=nfloat)
    
  for k1 in range(row_len): # Unreadable calculation
    i = row[k1]
    j = col[k1]
    dist = euclidian(qx[i], qy[i], qx[j], qy[j])
    ### Calculations on 
    # Elastic      
    dEVxdqx[k1] = (-k/m[i]*(qx[i]-qx[j])**2/dist**2 \
                     -k/m[i]*(dist-L0[k1])*(1/dist-(qx[i]-qx[j])**2/dist**3))
    dEVydqy[k1] = (-k/m[i]*(qy[i]-qy[j])**2/dist**2 \
                            -k/m[i]*(dist-L0[k1])*(1/dist-(qy[i]-qy[j])**2/dist**3))
    dEVxdqy[k1] = (-k/m[i]*(qy[i]-qy[j])*(qx[i]-qx[j])/dist**2 \
                        -k/m[i]*(dist-L0[k1])*(-(qy[i]-qy[j])*(qx[i]-qx[j])/dist**3))
    dEVydqx[k1] = (-k/m[i]*(qy[i]-qy[j])*(qx[i]-qx[j])/dist**2 \
                       -k/m[i]*(dist-L0[k1])*(-(qy[i]-qy[j])*(qx[i]-qx[j])/dist**3))
                           
    # Resistance (plastic)      
    dVxdpx[k1] = (-v/(m[i]*dist**2)*(qx[i]-qx[j])**2)
    dVxdpy[k1] = (-v/(m[i]*dist**2)*(qx[i]-qx[j])*(qy[i]-qy[j]))
    dVydpx[k1] = (-v/(m[i]*dist**2)*(qx[i]-qx[j])*(qy[i]-qy[j]))
    dVydpy[k1] = (-v/(m[i]*dist**2)*(qy[i]-qy[j])**2)
    
    dEVxdqx[k1] += (2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qx[i]-qx[j])**2 \
             -v/(m[i]*dist**2)*(dqx[i]-dqx[j])*(qx[i]-qx[j]))
    dEVxdqy[k1] += (2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qy[i]-qy[j])*(qx[i]-qx[j]) \
              -v/(m[i]*dist**2)*(dqy[i]-dqy[j])*(qx[i]-qx[j]))
    dEVydqx[k1] += (2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qx[i]-qx[j])*(qy[i]-qy[j]) \
              -v/(m[i]*dist**2)*(dqx[i]-dqx[j])*(qy[i]-qy[j]))
    dEVydqy[k1] += (2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qy[i]-qy[j])**2 \
                -v/(m[i]*dist**2)*(dqy[i]-dqy[j])*(qy[i]-qy[j]))

    if reduced and j in fix_indexes:
      continue
    # Elastic
    dEVxdqx[k2 + row_len] = (-k/m[i]*(qx[j]-qx[i])/dist*(qx[i]-qx[j])/dist \
                   -k/m[i]*(dist-L0[k1])*(-1/dist-(qx[i]-qx[j])*(qx[j]-qx[i])/dist**3))
    dEVydqy[k2 + row_len] = (-k/m[i]*(qy[j]-qy[i])/dist*(qy[i]-qy[j])/dist \
                           -k/m[i]*(dist-L0[k1])*(-1/dist-(qy[i]-qy[j])*(qy[j]-qy[i])/dist**3))
    dEVxdqy[k2 + row_len] = (-k/m[i]*(qy[j]-qy[i])/dist*(qx[i]-qx[j])/dist \
                        -k/m[i]*(dist-L0[k1])*(-(qx[i]-qx[j])*(qy[j]-qy[i])/dist**3))
    dEVydqx[k2 + row_len] = (-k/m[i]*(qx[j]-qx[i])/dist*(qy[i]-qy[j])/dist \
                       -k/m[i]*(dist-L0[k1])*(-(qy[i]-qy[j])*(qx[j]-qx[i])/dist**3))
                           
    # Resistance (plastic)
    dVxdpx[k2 + row_len] = (v/(m[i]*dist**2)*(qx[i]-qx[j])**2)
    dVxdpy[k2 + row_len] = (v/(m[i]*dist**2)*(qx[i]-qx[j])*(qy[i]-qy[j]))
    dVydpx[k2 + row_len] = (v/(m[i]*dist**2)*(qx[i]-qx[j])*(qy[i]-qy[j]))
    dVydpy[k2 + row_len] = (v/(m[i]*dist**2)*(qy[i]-qy[j])**2)
    
    dEVxdqx[k2 + row_len] += (-2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qx[i]-qx[j])**2 \
            +v/(m[i]*dist**2)*(dqx[i]-dqx[j])*(qx[i]-qx[j]))
    dEVxdqy[k2 + row_len] += (2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qy[j]-qy[i])*(qx[i]-qx[j]) \
              +v/(m[i]*dist**2)*(dqy[i]-dqy[j])*(qx[i]-qx[j]))
    dEVydqx[k2 + row_len] += (2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qx[j]-qx[i])*(qy[i]-qy[j]) \
              +v/(m[i]*dist**2)*(dqx[i]-dqx[j])*(qy[i]-qy[j]))
    dEVydqy[k2 + row_len] += (-2*v/(m[i]*dist**4)*((qx[i]-qx[j])*(dqx[i]-dqx[j]) + (qy[i]-qy[j])*(dqy[i]-dqy[j]))*(qy[i]-qy[j])**2 \
                +v/(m[i]*dist**2)*(dqy[i]-dqy[j])*(qy[i]-qy[j]))
    k2 += 1
  
  if reduced:
    row = list(map(reduced_index, row))
    row2 = list(map(reduced_index, row2))
    col2 = list(map(reduced_index, col2))
    indices = (row + row2, row + col2)
    N = n-2
  else:
    row, col = list(row), list(col)
    indices = (row + row, row + col)
    N = n

  dqdq = sparse.eye(2*N, format='csr')
  shape = (N, N)
  empty = sparse.csr_matrix((2*N, 2*N))
  
  if not reduced:
    for i in fix_indexes:
      dqdq[i, i] = 0
      dqdq[i+n, i+n] = 0
  
  dEVxdqx2 = sparse.coo_matrix((dEVxdqx, indices), shape=shape)
  dEVydqy2 = sparse.coo_matrix((dEVydqy, indices), shape=shape)
  dEVxdqy2 = sparse.coo_matrix((dEVxdqy, indices), shape=shape)
  dEVydqx2 = sparse.coo_matrix((dEVydqx, indices), shape=shape)
  dEVdq = sparse.vstack([sparse.hstack([dEVxdqx2, dEVxdqy2]), sparse.hstack([dEVydqx2, dEVydqy2])]).tocsr()

  dVxdpx2 = sparse.coo_matrix((dVxdpx, indices), shape=shape)
  dVydpy2 = sparse.coo_matrix((dVydpy, indices), shape=shape)
  dVxdpy2 = sparse.coo_matrix((dVxdpy, indices), shape=shape)
  dVydpx2 = sparse.coo_matrix((dVydpx, indices), shape=shape)
  dVdp = sparse.vstack([sparse.hstack([dVxdpx2, dVxdpy2]), sparse.hstack([dVydpx2, dVydpy2])]).tocsr()
  
  jacobian = sparse.vstack([sparse.hstack([dVdp, dEVdq]), sparse.hstack([dqdq, empty])])
  
  
  if with_inverse:
    jacobian_inverse = Jacobian_Inverse(dEVdq, dVdp, cholesky=with_cholesky, impact_index=reduced_index(impact_index) if reduced else impact_index)
  else:
    jacobian_inverse = None
  return jacobian, jacobian_inverse
