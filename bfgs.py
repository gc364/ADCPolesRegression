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

def apply_stages(poles,zeros,X_spectra,Y_spectra,frequencies,coeffs_per_stage):
    stages = [] #   The poles and zeros for each stage
    running = 0
    
    for nc in coeffs_per_stage:
        stages.append((poles[running:running+nc],zeros[running:running+nc]))
       
        running+=nc
    X_i = X_spectra
    for i,stage in enumerate(stages):
      
        if i<len(stages)-1:
            X_i = g_scipy(stage[0],stage[1],X_i,Y_spectra,frequencies,True)
        else:
            l = g_scipy(stage[0],stage[1],X_i,Y_spectra,frequencies,False)
    return l

def objective_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,scipy=False,coeffs_per_stage=None,set_poles = None):

    zeros = m[:nzeros//2] +  m[nzeros//2:nzeros]*1j

    if set_poles is None:
        poles = m[nzeros:(3*nzeros)//2]+m[(3*nzeros)//2:]*1j
    else:
        poles = set_poles

    if coeffs_per_stage is not None:
        l = -apply_stages(poles,zeros,X_spectra,Y_spectra,frequencies,coeffs_per_stage)
    elif scipy == False:
        l  = -g(poles,zeros,X_spectra,Y_spectra,frequencies)
    else:
        l = -g_scipy(poles,zeros,X_spectra,Y_spectra,frequencies)
    
    return np.sqrt(l.__abs__()@l.__abs__().T)

def jac_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,scipy=False,coeffs_per_stage=None,set_poles=None):
    g = np.zeros_like(m)
    eps = 1e-5
    for i in range(m.shape[0]):
        mn = m.copy()
        mn[i]-= eps
        lower = objective_lbfgs(mn,nzeros,X_spectra,Y_spectra,frequencies,scipy=scipy,coeffs_per_stage=coeffs_per_stage,set_poles=set_poles)
        mp = m.copy()
        mp[i]+= eps
        upper = objective_lbfgs(mp,nzeros,X_spectra,Y_spectra,frequencies,scipy=scipy,coeffs_per_stage=coeffs_per_stage,set_poles=set_poles)

        g[i] = (upper-lower)/eps

    return g

def callback_lbfgs(intermediate_result):
    print(f'Iteration Loss: {intermediate_result.fun}')
    return




def main_lbfgs():

    #   5th order sinc filter (boxcar thing)
    #   followed by a min phase FIR 4 stage FIR, low pass only  


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
    
    X_spectra+= X_spectra1#+X_spectra2
    Y_spectra+=Y_spectra1#+Y_spectra2
    #frequencies = np.concatenate([frequencies,frequ])
    pulses = np.vstack([pulses,pulses1])


    #   Try applying the sinc filter to the input so we only need to estimate the FIR
    fig,ax  =plt.subplots()
    ax.plot(frequencies,X_spectra[50].real.__abs__(),'k-',label='Pre SINC')
    for i,X in enumerate(X_spectra):
        X_spectra[i]  =apply_sinc_filter(X,256,frequencies)


    ax.plot(frequencies,X_spectra[50].real.__abs__(),'r--',label='Post SINC')
    ax.semilogy()
    ax.legend()
    plt.savefig('test.png')
    #exit()
    #   We might need to change how we optimise stuff. As of now we only have 1 stage, but really we need to have 4 in one
    #   go bearing in mind it becomes unstable past 70 poles

    alpha = 1
    nepochs = 400
    new_optimise = True

    if new_optimise:
        zeros,poles = create_fir_filter_PZs()
        coeffs_per_stage = [35]
        nz = sum(coeffs_per_stage)*2     #   This is the TOTAL number of zeros, but we're only regressing for half of them

        #   Fixed poles? and if so what, just put the poles in the upper complex plane, split into real and imaginary
        #   This is useful for reducing the complexity
        set_poles = Polynomial(np.ones_like(nz)).roots()    


        ###########################################
        #       BOUNDS      #
        #   If these are all within the unit circle its a minimum phase filter
        #   For a linear phase filter we need to look for 4-tuples of roots.
        #   that is each root in the upper plane, z,say, must have a conjugate z' (as before),
        #   as well as a reciprocal conjugate pair, that is 1/z and 1/z'.
        #   
        #   Essentially these are the most important but for choosing what type of filter we get out
        #   We can partition these into regressing for multiple filter stages


        zeros_real_max = 1#zeros.real.max()
        zeros_imag_max = 1#zeros.imag.max()
        poles_real_max = 1#zeros.real.max()
        poles_imag_max = zeros.imag.max()


        zeros_real_min = -1#zeros.real.min()
        zeros_imag_min = zeros.imag.min()
        poles_real_min = -1#zeros.real.min()
        poles_imag_min = zeros.imag.min()

        start_max = 2000

        poles = np.concatenate([np.random.uniform(poles_real_min,poles_real_max,nz//2),np.random.uniform(0,poles_imag_max,nz//2)])#-np.linspace(0.1,100,nz//2)+np.linspace(0j,100j,nz//2)
        zeros = np.concatenate([np.random.uniform(zeros_real_min,zeros_real_max,nz//2),np.random.uniform(0,zeros_imag_max,nz//2)])#np.zeros(nz//2)#np.linspace(-100,100,nz)

        #zeros,poles = create_fir_filter_PZs()

        if set_poles is not None:
            poles=set_poles
        
            m0 = np.concatenate([zeros])
            
            bounds_zeros = [(-1,1) for _ in range(nz//2)]  #Real
            bounds_zeros+=[(0,1) for _ in range(nz//2)]  #Imag

            bounds = bounds_zeros 
        else:
            m0 = np.concatenate([zeros,poles])
        
            bounds_poles = [(-1,1) for _ in range(nz//2)]  #Real
            bounds_poles+=[(0,1) for _ in range(nz//2)]  #Imag
            bounds_zeros = [(-1,1) for _ in range(nz//2)]  #Real
            bounds_zeros+=[(0,1) for _ in range(nz//2)]  #Imag


            bounds = bounds_zeros + bounds_poles

        ######################################


        res = minimize(
                lambda m:objective_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles),
                x0=m0,
                jac = lambda m: jac_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles=set_poles),
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
        if set_poles is None:
            poles_final = poles
        else:
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

    #Choose a sampel of low, mid and high Frequebcy data
    fig,ax = plt.subplots(3,layout='constrained')

    
    for y,d,axs in zip([Y_spectra[0],Y_spectra[50],Y_spectra[100]],[data_reconst[0],data_reconst[50],data_reconst[100]],ax):
        axs.plot(frequencies,10**y,'k-',label='Observations')
        axs.plot(frequencies,10**d,'r--',label='Synthetics')
        axs.loglog()
        
        axs.set_ylabel(r'Log(Amp)')
    ax[-1].set_xlabel('Iteration')
    fig.savefig(f'{FIGPATH}/data_fit.png')
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