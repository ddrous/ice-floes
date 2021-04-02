import numpy as np
import numpy.linalg as nplin
import scipy as scipy

## Les constantes
m = 1.0
m_ = 1.0

k = 2.0
k_ = 1.0

mu = 2.0
mu_ = 1.0


E = np.array([[0,0,1,0], 
              [0,0,0,1], 
              [(k-k_)/(m+m_), k_/(m+m_), (mu-mu_)/(m+m_), mu_/(m+m_)], 
              [k_/m_, -k_/m_, mu_/m_, -mu_/m_]])

print("E:\n", E)

Eigs = nplin.eig(E)
print(Eigs)


## Voir COLAB pour la SUITE !!!