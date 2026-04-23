import numpy as np
import scipy
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path
import tqdm
from numpy.polynomial import Polynomial
from filter_coeffs import *
from scipy.optimize import minimize
from shared import *


FIGPATH = 'figures/gn'



def create_G(poles,zeros,X_spectra,Y_spectra,frequencies):
    """This creates the Jacobian"""
    G = np.zeros(shape=(len(X_spectra),poles.shape[0]+zeros.shape[0]),dtype=np.complex128)
    eps = 1e-12
    epsi = eps*1j
    nz = zeros.shape[0]
    for m in range(zeros.shape[0]):
        bp = zeros.copy()
        bp[m] +=eps
        bn = zeros.copy()
        bn[m] -=eps
        
        poles1,bp = sort_model_vector(np.concatenate([bp,poles]),nz,poles_and_zeros=True)
        poles1,bn = sort_model_vector(np.concatenate([bn,poles]),nz,poles_and_zeros=True)
        diff = (g(poles1,bp,X_spectra,Y_spectra,frequencies)-g(poles1,bn,X_spectra,Y_spectra,frequencies))/eps
        G[:,m] = diff

    for m in range(poles.shape[0]):
        
        ap = poles.copy()
        ap[m] +=eps
        an = poles.copy()
        an[m] -=eps

        ap,zeros1 = sort_model_vector(np.concatenate([zeros,ap]),nz,poles_and_zeros=True)
        an,zeros1 = sort_model_vector(np.concatenate([zeros,an]),nz,poles_and_zeros=True)

        diff = (g(ap,zeros1,X_spectra,Y_spectra,frequencies)-g(an,zeros1,X_spectra,Y_spectra,frequencies))/eps

        G[:,zeros.shape[0]+m] = diff
    return G


def hess(G):
    return G.T@G

def jac(G,m,nzeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray):

    poles,zeros = sort_model_vector(m,nzeros,poles_and_zeros=True)
    
    return G.T@(-g(poles,zeros,X_spectra,Y_spectra,frequencies))


def model_update(m_last,nzeros,Qm,X_spectra,Y_spectra,frequencies):
    zeros = m_last[:nzeros]
    poles = m_last[nzeros:]
    G = create_G(poles,zeros,X_spectra,Y_spectra,frequencies)

    H = hess(G)
    s = jac(G,m_last,nzeros,X_spectra,Y_spectra,frequencies)
    m_post  = m_last+ np.linalg.solve(H+Qm,s)
    return m_post

def sort_model_vector(m,nzeros,poles_and_zeros=False):
    """Sorts a real partitioend model vector into a compelx vector"""
    zeros = m[:nzeros//2] +  m[nzeros//2:nzeros]*1j
    poles = m[nzeros:(3*nzeros)//2]+m[(3*nzeros)//2:]*1j
    if poles_and_zeros:
        return poles,zeros
    return np.concatenate([zeros,poles])

def optimise(nz,X_spectra,Y_spectra,frequencies,nepochs=100,alpha=1e-2):
    """Optimise for the poles and zeros"""
    poles  = np.concatenate([np.random.uniform(-2000,0,nz//2),np.random.uniform(0,2000,nz//2)])
    zeros = np.concatenate([np.random.uniform(-2000,2000,nz//2),np.random.uniform(0,2000,nz//2)])

    #   Zeros First, Poles second
    m0 = np.concatenate([zeros,poles])
    Qm = np.eye(poles.shape[0]+zeros.shape[0])*alpha
    mi = m0
    losses = []
    for e in tqdm.tqdm(range(nepochs),desc = 'Optimising'):
        mi1 = model_update(mi,nz,Qm,X_spectra,Y_spectra,frequencies)

        poles_i,zeros_i = sort_model_vector(mi1,nz,poles_and_zeros=True)
        loss  = g(poles_i,zeros_i,X_spectra,Y_spectra,frequencies)
        print(sum(loss.__abs__()**2)**0.5)
        losses.append(sum(loss.__abs__()**2)**0.5)
        mi = mi1
        mask = poles_i.real > 0
        poles_i[mask] = 0 +1j*poles_i[mask].imag
        mi[nz:3*(nz)//2] = poles_i.real
        mi[3*(nz)//2:] = poles_i.imag
        
    m_post = mi
    return m_post,losses

def calculate_C_posterior(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,alpha):
    G = create_G(poles_final,zeros_final,X_spectra,Y_spectra,frequencies)
    Qm = alpha*np.eye(G.shape[1])
    return np.linalg.inv(hess(G)+Qm)


def determine_nz(nzs,X_spectra,Y_spectra,frequencies,nepochs=100,alpha=1e-2):
    """
    Determines the best order of polynomial via an L-curve
    """
    #   The Determination of optimum number of poles
    m_norms = []
    d_norms = []
    for nz in nzs:
        m_post,losses = optimise(nz,X_spectra,Y_spectra,frequencies,nepochs,alpha)
        d_norms.append(losses[-1])
        m_norms.append(m_post@m_post.T)
    log_d_norms = np.log10(d_norms)
    log_m_norms = np.log10(m_norms)
    fig,ax =  plt.subplots()
    ax.plot(log_m_norms,log_d_norms)
    ax.set_ylabel(r'||d||')
    ax.set_xlabel(r'||m||')
    plt.savefig(f'{FIGPATH}/Nz.png')

    grad = np.diff(log_d_norms)/np.diff(log_m_norms)
    min_ind = np.argmin(grad)
    nz_opt = (nzs[min_ind+1]-nzs[min_ind])//2
    return int(nz_opt)

def example_filter():
    coeffs = STAGE1_LINEAR
    transfer_function = Polynomial(coeffs,domain=[0,6000])
    transfer_function.roots()
    return 


def main():
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_0')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra,Y_spectra,frequencies  = load_xy(pulses_path,nc_path)
    alpha = .1
    nepochs = 100
    new_optimise = True
    if new_optimise:
        nz = 70 
        m_post,losses = optimise(nz,X_spectra,Y_spectra,frequencies,nepochs,alpha)
        poles_final,zeros_final = sort_model_vector(m_post,nz,poles_and_zeros=True)

        pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])])
        np.save(f'{FIGPATH}/PolesandZeros.npy',pandz)
        C_post = calculate_C_posterior(m_post[nz:],m_post[:nz],X_spectra,Y_spectra,frequencies,alpha)
        np.save(f'{FIGPATH}/PosteriorCovariance.npy',C_post)
        data_reconst = g(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,True)

        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
    else:
        C_post = np.load(f'{FIGPATH}/PosteriorCovariance.npy')
        pandz =  np.load(f'{FIGPATH}/PolesandZeros.npy')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
        nz = zeros_final.shape[0]
        data_reconst = g(poles_final[nz//2:],zeros_final[:nz//2],X_spectra,Y_spectra,frequencies,True)

    fig,ax = plt.subplots(2,layout='constrained')
    H = calculate_transfer_function(poles_final,zeros_final,2*np.pi*frequencies)

    ax[0].plot(frequencies,H.real/(2*np.pi))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(r'$\mathfrak{R}$')
    ax[1].set_ylabel(r'$\mathfrak{I}$')
    ax[1].plot(frequencies,H.imag)
    ax[0].axvline(x=250)
    ax[1].axvline(x=250)

    ax[0].loglog()
    ax[1].loglog()
    
    plt.savefig(f'{FIGPATH}/Transfer_function.png')
    plt.close()

    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real,poles_final.imag,'x',label='Poles')
    ax.plot(zeros_final.real,zeros_final.imag,'o',label='Zeros')
    ax.set_ylabel(r'$\mathfrak{Im} (rad/s)$')
    ax.set_xlabel(r'$\mathfrak{Re} (rad/s)$')
    ax.grid()
    plt.savefig(f'{FIGPATH}/PoleandZeros.png')
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real/(2*np.pi),poles_final.imag/(2*np.pi),'x',label='Poles')
    ax.plot(zeros_final.real/(2*np.pi),zeros_final.imag/(2*np.pi),'o',label='Zeros')
    ax.set_ylabel(r'$\mathfrak{Im} (Hz)$')
    ax.set_xlabel(r'$\mathfrak{Re} (Hz)$')
    ax.grid()
    plt.savefig(f'{FIGPATH}/PoleandZerosHz.png')


    fig,(ax,ax1) = plt.subplots(2,layout='constrained')
    if new_optimise:
        ax.plot(losses)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$||\Delta d||_{2}^{2}$')
    for y,d in zip(Y_spectra,data_reconst):
        ax1.plot(frequencies,y,'r')
        ax1.plot(frequencies,d,'k')
    fig.savefig(f'{FIGPATH}/losses.png')
    plt.close()
    fig,ax = plt.subplots(layout='constrained')
    im = ax.imshow(C_post.real)
    plt.colorbar(im,ax=ax,label='Posterior Variance')
    plt.savefig(f'{FIGPATH}/PosteriorCovar.png',dpi=256)

if __name__ == '__main__':
    main()