import numpy as np
from scipy.integrate import odeint
from scipy.integrate import quad


def integrate(m=1.0, m_=1.0, k=3.0, k_=22.0, mu=6.0, mu_=2.0, v0=0, v_0=1.8, N=1000, tminus=0.0, tplus=5.0):
    E = np.array([[0,0,0, 1.0,0, 0], 
                  [0,0,0,0, 1.0, 0], 
                  [0,0,0,0, 0, 1.0], 
                  [-k/m, k/m, 0, -mu/m, mu_/m, 0], 
                  [k/(m+m_), (-k-k_)/(m+m_), k/(m+m_),  mu_/(m+m_), (-mu-mu_)/(m+m_), mu_/(m+m_)], 
                  [0, k_/m_, -k_/m_,  0, mu_/m_, -mu_/m_]])
    vf = (m*v0 - m_*v_0)/(m + m_)
    Y0 = np.array([0,0, 0, vf, vf, vf])
    t = np.linspace(tminus, tplus, N+1)
    
    def model(Y, t):
        return E @ Y
    Y = odeint(model, Y0, t)

    I = k*(Y[:,0]-Y[:,1]) + mu*(Y[:,3]-Y[:,4]) - k_*(Y[:,1]-Y[:,2]) - mu_*(Y[:,4]-Y[:,5])

    return np.trapz(I, x=t)

