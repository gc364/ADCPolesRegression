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


#   TODO:   All these need to be constrained in polar coordinates not cartesian
#           Easiest to make the model vector in polar then convert to cartesian for
#           computations then back again. This means the bounds will be for r and phi

def polar_to_cartesian(poles,zeros):
    nz = zeros.shape[0]

    r_z = zeros[:nz]
    phi_z  = zeros[nz:]

    ret_zeros = np.zeros_like(zeros)

    real_zeros,imag_zeros = r_z*np.cos(phi_z),r_z*np.sin(phi_z)
    ret_zeros[:] = np.concatenate([real_zeros,imag_zeros])

    r_p = poles[:nz]
    phi_p  = poles[nz:]

    ret_poles = np.zeros_like(poles)

    real_poles,imag_poles = r_p*np.cos(phi_p),r_p*np.sin(phi_p)
    ret_poles[:] = np.concatenate([real_poles,imag_poles])

    return ret_poles, ret_zeros 

def cartesian_to_polar(poles,zeros):
    nz = zeros.shape[0]

    real_z = zeros[:nz]
    imag_z  = zeros[nz:]

    ret_zeros = np.zeros_like(zeros)

    r_z,phi_z = np.sqrt(real_z**2 + imag_z**2),np.arctan2(imag_z,real_z)
    ret_zeros[:] = np.concatenate([r_z,phi_z])

    real_p = poles[:nz]
    imag_p  = poles[nz:]

    ret_poles = np.zeros_like(poles)

    real_poles,imag_poles = np.sqrt(real_p**2 + imag_p**2),np.arctan2(imag_p,real_p)
    ret_poles[:] = np.concatenate([real_poles,imag_poles])

    return ret_poles, ret_zeros 



def objective_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,
                    scipy=False,coeffs_per_stage=None,set_poles=None,phase='MIN'):

    #   changing to cartesian from polar
    zeros = m[:nzeros//2]*np.exp(1j* m[nzeros//2:nzeros]) #+  m[nzeros//2:nzeros]*1j

    if phase == 'LINEAR':
        mask = zeros.real >1
        zeros = np.concatenate([zeros,1/zeros[mask]])

    if set_poles is None:
        poles = m[nzeros:(3*nzeros)//2]*np.exp(1j*m[(3*nzeros)//2:])#+m[(3*nzeros)//2:]*1j
    else:
        poles = set_poles
        poles = poles[0:nzeros//2]*np.exp(1j*poles[(nzeros)//2:])
    
    if coeffs_per_stage is not None:
        l = -apply_stages(poles,zeros,X_spectra,Y_spectra,frequencies,coeffs_per_stage)
    elif scipy == False:
        l  = -g(poles,zeros,X_spectra,Y_spectra,frequencies)
    else:
        l = -g_scipy(poles,zeros,X_spectra,Y_spectra,frequencies)
    
    return np.sqrt(l.__abs__()@l.__abs__().T)

def jac_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,scipy=False,coeffs_per_stage=None,set_poles=None,phase='MIN'):

    #   I think we need to scale the angular part of g by r (the top half of g)

    g = np.zeros_like(m)
    eps = 1e-5
    
    for i in range(m.shape[0]):
        mn = m.copy()
        mn[i]-= eps
        lower = objective_lbfgs(mn,nzeros,X_spectra,Y_spectra,frequencies,
                                scipy=scipy,coeffs_per_stage=coeffs_per_stage,
                                set_poles=set_poles,phase=phase)
        mp = m.copy()
        mp[i]+= eps
        upper = objective_lbfgs(mp,nzeros,X_spectra,Y_spectra,frequencies,
                                scipy=scipy,coeffs_per_stage=coeffs_per_stage,
                                set_poles=set_poles,phase=phase)

        g[i] = (upper-lower)/eps

    g[m.shape[0]//2:] *= 1/g[:m.shape[0]//2]

    return g

def callback_lbfgs(intermediate_result):
    print(f'Value: {intermediate_result.x}')
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
    pulses = pulses[1:,:]
 
    if sinc_dec is not None:
        for i,X in enumerate(X_spectra):
            for s in sinc_dec:
                X =apply_sinc_filter(X,s,frequencies)
            X_spectra[i] = X
    
    return X_spectra,Y_spectra,frequencies,pulses

def initialise_model(nz:int,bounds:dict,set_poles,unit_circle=True):
    if not unit_circle:
        if bounds.phase == 'MIN' or bounds.phase is None:
            zeros = np.concatenate([
                                    np.random.uniform(bounds.zeros_r_min,bounds.zeros_r_max,nz//2),
                                    np.random.uniform(bounds.zeros_phi_min,bounds.zeros_phi_max,nz//2)
                                    ])
        elif bounds.phase=='LINEAR':
            zeros = np.concatenate([
                                    np.random.uniform(bounds.zeros_r_min,bounds.zeros_r_max,nz//4),
                                    np.random.uniform(bounds.zeros_phi_min,bounds.zeros_phi_max,nz//4)
                                    ])
    else:
        if bounds.phase == 'MIN' or bounds.phase is None:
            phi = np.linspace(0,np.pi,nz//2)
            r = np.ones(nz//2)-0.3
            zeros = np.concatenate([r,phi])
        elif bounds.phase=='LINEAR':
            phi = np.linspace(0,np.pi,nz//4)
            r = np.ones(nz//4)-0.3
            zeros = np.concatenate([r,phi])
    
    if set_poles is None:
        poles = np.concatenate([
                                np.random.uniform(bounds.poles_r_min,bounds.poles_r_max,nz//2),
                                np.random.uniform(bounds.poles_phi_min,bounds.poles_phi_max,nz//2)
                                ])
        m0 = np.concatenate([zeros,poles])
        return m0

    return zeros
    
def get_bounds(phase:str):
    bounds = {'zeros_r_min':None,
              'zeros_r_max':None,
              'zeros_phi_min':None,
              'zeros_phi_max':None,
              'poles_r_min':None,
              'poles_r_max':None,
              'poles_phi_min':None,
              'poles_phi_max':None,
              'phase':None
              }
    
    bounds = SimpleNamespace(**bounds)

    if phase == 'MIN':
        bounds.zeros_r_max = 1
        bounds.zeros_phi_max = np.pi
        bounds.poles_r_max = 1
        bounds.poles_phi_max = np.pi


        bounds.zeros_r_min = 0
        bounds.zeros_phi_min = 0
        bounds.poles_r_min = 0
        bounds.poles_phi_min = 0
        bounds.phase = 'MIN'
    elif phase is None:
        bounds.zeros_r_max = None
        bounds.zeros_phi_max = None
        bounds.poles_r_max = None
        bounds.poles_phi_max = None

        bounds.zeros_r_min = 0
        bounds.zeros_phi_min = 0
        bounds.poles_r_min = 0
        bounds.poles_phi_min = 0
    elif phase == 'LINEAR':
        bounds.zeros_r_max = 10
        bounds.zeros_phi_max = np.pi
        bounds.poles_r_max = 10
        bounds.poles_phi_max = np.pi

        bounds.zeros_r_min = 0
        bounds.zeros_phi_min = 0
        bounds.poles_r_min = 0
        bounds.poles_phi_min = 0
        bounds.phase = 'LINEAR'
    else:
        raise ValueError('Unrecognised Type')
    return bounds
    

def sort_bounds(bounds,nz,set_poles):
    if bounds.phase == 'MIN':
        bounds_zeros = [(bounds.zeros_r_min,bounds.zeros_r_max) for _ in range(nz//2)]  #Real
        bounds_zeros+=[(bounds.zeros_phi_min,bounds.zeros_phi_max) for _ in range(nz//2)]  #Imag
    elif bounds.phase =='LINEAR':
        bounds_zeros = [(bounds.zeros_r_min,bounds.zeros_r_max) for _ in range(nz//4)]  #Real
        bounds_zeros+=[(bounds.zeros_phi_min,bounds.zeros_phi_max) for _ in range(nz//4)]  #Imag
    
    if set_poles is not None:
        return bounds_zeros
    bounds_poles = [(bounds.poles_r_min,bounds.poles_r_max) for _ in range(nz//2)]  #Real
    bounds_poles+=[(bounds.poles_phi_min,bounds.poles_phi_max) for _ in range(nz//2)]  #Imag

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
        if phase == 'MIN':
            set_poles = np.zeros(nz)
        elif phase == 'LINEAR':
            set_poles = np.zeros(nz//2)
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
        bounds = get_bounds(phase)

    else:
        raise ValueError('Unknown Type/Phase pair')
    m0 = initialise_model(nz,bounds,set_poles)
    ret_bounds = sort_bounds(bounds,nz,set_poles)
    return m0,ret_bounds,set_poles,nz


def sort_mpost(m_post,nz,set_poles,phase):
   

    if phase=='MIN' or phase == None:
        if set_poles is not None:
                poles_final = set_poles[:nz//2]*np.exp(1j*set_poles[nz//2:]) #+ 1j*set_poles[nz//2:]
        else:
            poles_final = m_post[nz:(3*nz)//2]*np.exp(1j*m_post[(3*nz)//2:])#+1j*m_post[(3*nz)//2:]

        zeros_final = m_post[:nz//2]*np.exp(1j*m_post[nz//2:nz]) #+ 1j*m_post[nz//2:nz]


        pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])])
        return pandz
    

    elif phase=='LINEAR':
        if set_poles is not None:
            poles_final = set_poles[:nz//4]*np.exp(1j*set_poles[nz//4:])
        else:
            poles_final = m_post[nz//2:(3*nz)//4]*np.exp(1j*m_post[(3*nz)//4:])

        zeros_final = m_post[:nz//4]*np.exp( 1j*m_post[nz//4:nz//2])

        
        mask = zeros_final.real >1
        zeros_final = np.concatenate([zeros_final,1/zeros_final[mask]])
        pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),
                                 np.concatenate([zeros_final,np.conj(zeros_final)])])
        return pandz


def main_lbfgs(paths,coeffs_per_stage,f_type='FIR',f_phase='MIN',nepochs=100,new_optimise=True,ftol=1e-3,sinc_dec = [256]):
    FIGPATH = 'figures/bfgs'
    X_spectra,Y_spectra,frequencies,pulses = load_datasets(paths,sinc_dec)
    print('Datasets Loaded')
    if new_optimise:
        m0,bounds,set_poles,nz = get_system(f_type,f_phase,coeffs_per_stage)
        print('Got Settings')
        print('Optimsing...')
        if f_phase == 'LINEAR':
            nz_old = nz
            nz = nz//2

        res = minimize(
                lambda m:objective_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles,phase=f_phase),
                x0=m0,
                jac = lambda m: jac_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles,phase=f_phase),
                method='L-BFGS-B',  
                bounds=bounds,
                callback=callback_lbfgs,
                options={
                         'maxiter':nepochs,
                         'ftol':ftol
                         }   

                )
        print('Done!')
        print(res)
        m_post = res.x
        if f_phase == 'LINEAR':
            nz = nz_old
        pandz = sort_mpost(m_post,nz,set_poles,f_phase)
        np.save(f'{FIGPATH}/PolesandZeros.npy',pandz)
        print('Plotting')
        #out_plots(pandz,coeffs_per_stage,frequencies,X_spectra,Y_spectra,pulses,FIGPATH)
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]

    else:
        pandz =  np.load(f'{FIGPATH}/PolesandZeros.npy')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
     
    out_plots(pandz,coeffs_per_stage,frequencies,X_spectra,Y_spectra,pulses,FIGPATH)
    create_nice_figures(poles_final,zeros_final,X_spectra,Y_spectra,paths[0],pulses,frequencies,coeffs_per_stage)
    create_fir_filter_PZs()
    return

if __name__ == '__main__':
    #   Datasets to load
    paths = [
                Path('/run/media/obic/SSD/test/ADC_Filter_2'),
                Path('/run/media/obic/SSD/test/ADC_Filter_3'),
                #Path('/run/media/obic/SSD/test/ADC_Filter_4')    
            ]
    #   Number of roots in the top half of the compelx plane
    #   These will by conjugated, so the total order will be twice this number
    #   or in the case of linear phase will be 4 times this number from further symmetries
    #   The numbers need to be divisible by 4 for linear phase.

    #   We looking for a cascade of two linear filters followed by two min phase filters

    coeffs_per_stage = [32,32]
    phases_per_stage = ['LINEAR','LINEAR']
    #   Decimations for the sinc filters
    sinc_dec = [16,32,256]
    
    new_optimise = True
    main_lbfgs(paths,coeffs_per_stage,new_optimise=new_optimise,ftol=1e-5,sinc_dec = sinc_dec,f_phase='LINEAR',nepochs=100 )