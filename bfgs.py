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

FIGPATH = 'figures/bfgs'



def objective_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,scipy=False):

    zeros = m[:nzeros//2] +  m[nzeros//2:nzeros]*1j
    poles = m[nzeros:(3*nzeros)//2]+m[(3*nzeros)//2:]*1j
    if scipy == False:
        l  = -g(poles,zeros,X_spectra,Y_spectra,frequencies)
    else:
        l = -g_scipy(poles,zeros,X_spectra,Y_spectra,frequencies)
    return np.sqrt(l.__abs__()@l.__abs__().T)

def jac_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,scipy=False):
    g = np.zeros_like(m)
    eps = 1e-5
    for i in range(m.shape[0]):
        mn = m.copy()
        mn[i]-= eps
        lower = objective_lbfgs(mn,nzeros,X_spectra,Y_spectra,frequencies,scipy=scipy)
        mp = m.copy()
        mp[i]+= eps
        upper = objective_lbfgs(mp,nzeros,X_spectra,Y_spectra,frequencies,scipy=scipy)

        g[i] = (upper-lower)/eps

    return g

def callback_lbfgs(intermediate_result):
    print(f'Iteration Loss: {intermediate_result.fun}')
    return



def main_lbfgs():
    #   The log-spaced samples
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_2')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra,Y_spectra,frequencies  = load_xy(pulses_path,nc_path)
    pulses = load_pulses(pulses_path)
    #   The uniform random samples
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_3')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra1,Y_spectra1,_  = load_xy(pulses_path,nc_path)
    pulses1 = load_pulses(pulses_path)

    #   The harmonic samples
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_4')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra2,Y_spectra2,_  = load_xy(pulses_path,nc_path)
    pulses2 = load_pulses(pulses_path)
    
    X_spectra+= X_spectra1+X_spectra2
    Y_spectra+=Y_spectra1+Y_spectra2
    #frequencies = np.concatenate([frequencies,frequ])
    pulses = np.vstack([pulses,pulses1])


    alpha = 1
    nepochs = 100
    new_optimise = True
    if new_optimise:

        nz = 70 

        ###########################################
        #       BOUNDS      #
        #   If these are all within the unit circle its a minimum phase filter
        #   If symmetric across the real axis and all poles is top left quandrant its a FIR filter
        #   
        #   Essentially these are the most important but for choosing what type of filter we get out
        #   We can partition these into regressing for multiple filter stages
        poles = np.concatenate([np.random.uniform(-2000,0,nz//2),np.random.uniform(0,2000,nz//2)])#-np.linspace(0.1,100,nz//2)+np.linspace(0j,100j,nz//2)
        zeros = np.concatenate([np.random.uniform(-2000,2000,nz//2),np.random.uniform(0,2000,nz//2)])#np.zeros(nz//2)#np.linspace(-100,100,nz)
        m0 = np.concatenate([zeros,poles])
        bounds_poles = [(None,0) for _ in range(nz//2)]
        bounds_poles+=[(0,None) for _ in range(nz//2)]
        bounds_zeros = [(None,None) for _ in range(nz//2)]
        bounds_zeros+=[(0,None) for _ in range(nz//2)]
        bounds = bounds_zeros + bounds_poles

        ######################################


        res = minimize(
                lambda m:objective_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True),
                x0=m0,
                jac = lambda m: jac_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True),
                method='L-BFGS-B',  
                bounds=bounds,
                callback=callback_lbfgs,
                options={'disp':True,
                         'maxiter':nepochs,
                         'ftol':1e-3
                         }      
                )
        print(res)
        m_post = res.x
        
        poles_final = m_post[nz:(3*nz)//2]+1j*m_post[(3*nz)//2:]
        zeros_final = m_post[:nz//2] + 1j*m_post[nz//2:nz]
        pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])])
        np.save(f'{FIGPATH}/PolesandZeros.npy',pandz)

        data_reconst = g(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,True)

        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
    else:
       
        pandz =  np.load(f'{FIGPATH}/PolesandZeros.npy')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
        nz = zeros_final.shape[0]
        data_reconst = g_scipy(poles_final[nz//2:],zeros_final[:nz//2],X_spectra,Y_spectra,frequencies,True)

    fig,ax = plt.subplots(2,layout='constrained')
    H = calculate_transfer_function(poles_final,zeros_final,2*np.pi*frequencies)
    ax[0].plot(frequencies,H.real/(2*np.pi))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(r'$\mathfrak{R}$')
    ax[1].set_ylabel(r'$\mathfrak{I}$')
    ax[1].plot(frequencies,H.imag)
    ax[0].loglog()
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

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$||\Delta d||_{2}^{2}$')
    for y,d in zip(Y_spectra,data_reconst):
        ax1.plot(frequencies,10**y,'r')
        ax1.plot(frequencies,10**d,'k')
    ax1.loglog()
    fig.savefig(f'{FIGPATH}/losses.png')
    plt.close()

    w,h = scipy_frequency_response(poles_final,zeros_final,frequencies)
    fig,(ax,ax1) = plt.subplots(2)
    ax.plot(w,h.real)
    ax.plot(w,(h*X_spectra[0]).real)
    ax1.plot(w,h.imag)
    ax.loglog()

    plt.savefig(f'{FIGPATH}/scipyFreqz.png')

    fig,ax = plt.subplots()
    phase = np.angle(h)
    ax.plot(2*np.pi*w,phase)
    ax.set_xlabel(r'$\omega$ (radians)')
    ax.set_ylabel(r'$\Phi$ (radians)')
    ax.semilogx()
    plt.savefig(f'{FIGPATH}/phase_repsonse.png')

    create_nice_figures(poles_final,zeros_final,X_spectra,Y_spectra,nc_path,pulses,frequencies)
    return

if __name__ == '__main__':
    main_lbfgs()