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
from types import SimpleNamespace

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


def load_datasets(paths:list[Path],sinc_dec=None):
    """
    load and concatenate multiple datasets.
    
    Args:
        paths,
        sinc_dec
    Returns:
        X_spectra,Y_spectra,frequencies,pulses


    """
    X_spectra = []
    Y_spectra = []
    pulses = np.zeros(shape=(1,4))
    for workdir in paths:
        nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
        pulses_path = workdir.joinpath('processed/pulses.txt')
        X_spectra_i,Y_spectra_i,frequencies  = load_xy(pulses_path,nc_path)
        pulses_i = load_pulses(pulses_path)
        X_spectra+=X_spectra_i
        Y_spectra+=Y_spectra_i
        pulses = np.vstack([pulses,pulses_i])

    if sinc_dec is not None:
        for i,X in enumerate(X_spectra):
            X_spectra[i]  =apply_sinc_filter(X,sinc_dec,frequencies)
    
    return X_spectra,Y_spectra,frequencies,pulses

def initialise_model(nz:int,bounds:dict,set_poles):

    zeros = np.concatenate([
                            np.random.uniform(bounds.zeros_real_min,bounds.zeros_real_max,nz//2),
                            np.random.uniform(bounds.zeros_imag_min,bounds.zeros_imag_max,nz//2)
                            ])
    
    if set_poles is None:
        poles = np.concatenate([
                                np.random.uniform(bounds.poles_real_min,bounds.poles_real_max,nz//2),
                                np.random.uniform(bounds.poles_imag_min,bounds.poles_imag_max,nz//2)
                                ])
        m0 = np.concatenate([zeros,poles])
        return m0
    
    return zeros
    
def get_bounds(phase:str):
    bounds = {'zeros_real_min':None,
              'zeros_real_max':None,
              'zeros_imag_min':None,
              'zeros_imag_max':None,
              'poles_real_min':None,
              'poles_real_max':None,
              'poles_imag_min':None,
              'poles_imag_max':None,
              }
    
    bounds = SimpleNamespace(**bounds)

    if phase == 'MIN':
        bounds.zeros_real_max = 1
        bounds.zeros_imag_max = 1
        bounds.poles_real_max = 1
        bounds.poles_imag_max = 1


        bounds.zeros_real_min = -1
        bounds.zeros_imag_min = 0
        bounds.poles_real_min = -1
        bounds.poles_imag_min = 0
    elif phase is None:
        bounds.zeros_real_max = None
        bounds.zeros_imag_max = None
        bounds.poles_real_max = None
        bounds.poles_imag_max = None

        bounds.zeros_real_min = None
        bounds.zeros_imag_min = 0
        bounds.poles_real_min = None
        bounds.poles_imag_min = 0
    else:
        raise ValueError('Unrecognised Type')
    return bounds
    

def sort_bounds(bounds,nz,set_poles):
    bounds_zeros = [(bounds.zeros_real_min,bounds.zeros_real_max) for _ in range(nz//2)]  #Real
    bounds_zeros+=[(bounds.zeros_imag_min,bounds.zeros_imag_max) for _ in range(nz//2)]  #Imag
    if set_poles is not None:
        return bounds_zeros
    bounds_poles = [(bounds.poles_real_min,bounds.poles_real_max) for _ in range(nz//2)]  #Real
    bounds_poles+=[(bounds.poles_imag_min,bounds.poles_imag_max) for _ in range(nz//2)]  #Imag

    ret = bounds_zeros+bounds_poles
    return ret

def get_system(type:str,phase:str,coeffs_per_stage:list[int]):
    """
    Get the required bounds and settings for the specified parameters. 
    Return the initialiased poles and zeros and the bounds to be given to optimiser
    Args:
        type:   'FIR' or 'IIR',
        phase:   'MIN' or 'LINEAR' or None
    """
    nz = sum(coeffs_per_stage)*2 
    if type=='FIR':
        set_poles = np.zeros(nz)
        
    elif 'IIR':
        set_poles = None
        bounds = bounds
    else:
        raise ValueError('Unknown Filter Type')
    
    if phase is None :
        bounds = get_bounds(phase)
    elif phase == 'MIN' and type=='FIR':
        bounds = get_bounds(phase)
    elif phase == 'LINEAR':
        raise NotImplementedError('Not Done Yet')
        pass
    else:
        raise ValueError('Unknown Type/Phase pair')
    m0 = initialise_model(nz,bounds,set_poles)
    ret_bounds = sort_bounds(bounds,nz,set_poles)
    return m0,ret_bounds,set_poles,nz


def sort_mpost(m_post,nz,set_poles):
    if set_poles is not None:
            poles_final = set_poles[:nz//2] + 1j*set_poles[nz//2:]
    else:
        poles_final = m_post[nz:(3*nz)//2]+1j*m_post[(3*nz)//2:]

    zeros_final = m_post[:nz//2] + 1j*m_post[nz//2:nz]
    pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])])
    return pandz


def out_plots(pandz,frequencies,X_spectra,Y_spectra,pulses):

    poles_final,zeros_final = pandz[:,0],pandz[:,1]

    data_reconst = g_scipy(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,True)

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


    fig,ax = plt.subplots(3,layout='constrained')
    for y,d,axs,f in zip(
                        [Y_spectra[0],Y_spectra[50],Y_spectra[100]],
                        [data_reconst[0],data_reconst[50],data_reconst[100]],
                        ax,[pulses[0,1],pulses[50,1],pulses[100,1]]
                       ):
        axs.plot(frequencies,10**y,'k-',label='Observations')
        axs.plot(frequencies,10**d,'r--',label='Synthetics')
        axs.loglog()
        axs.set_title(rf'$f_x$ = {f}')
        
        axs.set_ylabel(r'Log(Amp)')
    ax[-1].set_xlabel('Iteration')
    fig.savefig(f'{FIGPATH}/data_fit.png')
    plt.close()


    w,h = scipy_frequency_response(poles_final,zeros_final,2*np.pi*frequencies)
    fig,(ax,ax1) = plt.subplots(2)
    ax.set_title(r'$\mathfrak{Re}(H(\omega))$')
    ax.plot(w,h.real.__abs__())
    ax1.set_title(r'$\mathfrak{Im}(H(\omega))$')
    ax1.plot(w,h.imag.__abs__())
    ax1.set_xlabel(r'$\omega$(rad/s)')
    ax.loglog()
    ax1.loglog()
    plt.savefig(f'{FIGPATH}/scipyFreqz.png')
    plt.close()

    fig,ax = plt.subplots()
    phase = np.angle(h)
    ax.plot(w,phase)
    ax.set_xlabel(r'$\omega$ (radians)')
    ax.set_ylabel(r'$\Phi$ (radians)')
    ax.semilogx()
    plt.savefig(f'{FIGPATH}/phase_repsonse.png')
    plt.close()
    return

def main_lbfgs(paths,coeffs_per_stage,f_type='FIR',f_phase='MIN',nepochs=100,new_optimise=True,ftol=1e-3):

    X_spectra,Y_spectra,frequencies,pulses = load_datasets(paths,256)
    
    if new_optimise:


        m0,bounds,set_poles,nz = get_system(f_type,f_phase,coeffs_per_stage)

        res = minimize(
                lambda m:objective_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles),
                x0=m0,
                jac = lambda m: jac_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles),
                method='L-BFGS-B',  
                bounds=bounds,
                callback=callback_lbfgs,
                options={'disp':True,
                         'maxiter':nepochs,
                         'ftol':ftol
                         }   

                )
        print(res)
        m_post = res.x
        pandz = sort_mpost(m_post,nz,set_poles)
        np.save(f'{FIGPATH}/PolesandZeros.npy',pandz)

        out_plots(pandz,frequencies,X_spectra,Y_spectra,pulses)
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]

    else:
        pandz =  np.load(f'{FIGPATH}/PolesandZeros.npy')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
     
     
    create_nice_figures(poles_final,zeros_final,X_spectra,Y_spectra,paths[0],pulses,frequencies)
    return

if __name__ == '__main__':

    paths = [
                Path('/run/media/obic/SSD/test/ADC_Filter_2'),
                Path('/run/media/obic/SSD/test/ADC_Filter_3'),
                Path('/run/media/obic/SSD/test/ADC_Filter_4')
                
                ]
    coeffs_per_stage = [35]
    
    main_lbfgs(paths,coeffs_per_stage,ftol=1e-2)