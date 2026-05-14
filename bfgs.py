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



def objective_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,phases_per_stage,
                    scipy=False,coeffs_per_stage=None,set_poles=None,phase='MIN'):

    #   changing to cartesian from polar
    zeros_list = []
    poles_list = []
  
    
    for i,(phase,nc) in enumerate(zip(phases_per_stage,coeffs_per_stage)):
        zeros = m[i*nc:(i+1)*nc]*np.exp(1j* m[(nzeros//2)+(i*nc):(nzeros//2)+((i+1)*nc)]) #+  m[nzeros//2:nzeros]*1j
        #print(f'Radius Start:   {i*nc} \n Radius End:   {(i+1)*nc}  \n Phase Start: {(nzeros//2)+(i*nc)}    \n  Phase end:  {(nzeros//2)+((i+1)*nc)}')
        if phase == 'LINEAR':
            zeros = np.concatenate([zeros,1/zeros])

        if set_poles is None:
            poles = m[nzeros:(3*nzeros)//2]*np.exp(1j*m[(3*nzeros)//2:])#+m[(3*nzeros)//2:]*1j
        else:
            poles = set_poles
            poles = poles[i*nc:(i+1)*nc]*np.exp(1j*poles[(nzeros//2)+(i*nc):(nzeros//2)+((i+1)*nc)])
        
        if zeros.shape[0]!= poles.shape[0]:
            to_pad = poles.shape[0]-zeros.shape[0]
            if to_pad >1:
                poles = poles[:poles.shape[0]-to_pad]
            elif to_pad <1:
                poles = np.concatenate([poles,np.zeros(-to_pad)])
        zeros_list.append(zeros)
        poles_list.append(poles)
  
    zeros = np.concatenate(zeros_list)
    poles = np.concatenate(poles_list)
  

    if coeffs_per_stage is not None:
        l = -apply_stages(poles,zeros,X_spectra,Y_spectra,frequencies,coeffs_per_stage)
    elif scipy == False:
        l  = -g(poles,zeros,X_spectra,Y_spectra,frequencies)
    else:
        l = -g_scipy(poles,zeros,X_spectra,Y_spectra,frequencies)
    
    return np.sqrt(l.__abs__()@l.__abs__().T)

def jac_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,phases_per_stage,scipy=False,coeffs_per_stage=None,set_poles=None,phase='MIN'):

    #   I think we need to scale the angular part of g by r (the top half of g)

    g = np.zeros_like(m)
    eps = 1e-5
    
    for i in range(m.shape[0]):
        mn = m.copy()
        mn[i]-= eps
        lower = objective_lbfgs(mn,nzeros,X_spectra,Y_spectra,frequencies,phases_per_stage,
                                scipy=scipy,coeffs_per_stage=coeffs_per_stage,
                                set_poles=set_poles,phase=phase)
        mp = m.copy()
        mp[i]+= eps
        upper = objective_lbfgs(mp,nzeros,X_spectra,Y_spectra,frequencies,phases_per_stage,
                                scipy=scipy,coeffs_per_stage=coeffs_per_stage,
                                set_poles=set_poles,phase=phase)

        g[i] = (upper-lower)/eps

    #g[m.shape[0]//2:] *= 1/g[:m.shape[0]//2]

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

def initialise_model(coeffs_per_stage:list[int],bounds_list:list[dict],set_poles,unit_circle=True):
    poles_list= []
    zeros_list_r = []
    zeros_list_phi = []
    for bounds,nz in zip(bounds_list,coeffs_per_stage):
        if not unit_circle:
            if bounds.phase == 'MIN' or bounds.phase is None:
                zeros_list_r.append(np.concatenate([
                                        np.random.uniform(bounds.zeros_r_min,bounds.zeros_r_max,nz),
                                        np.random.uniform(bounds.zeros_phi_min,bounds.zeros_phi_max,nz)
                                        ]))
            elif bounds.phase=='LINEAR':
                zeros_list_phi.append(np.concatenate([
                                        np.random.uniform(bounds.zeros_r_min,bounds.zeros_r_max,nz),
                                        np.random.uniform(bounds.zeros_phi_min,bounds.zeros_phi_max,nz)
                                        ]))
        else:
            if bounds.phase == 'MIN' or bounds.phase is None:
                phi = np.linspace(0,np.pi,nz)
                r = np.ones(nz)-0.3
                zeros_list_r.append(r)
                zeros_list_phi.append(phi)
            elif bounds.phase=='LINEAR':
                phi = np.linspace(0,np.pi,nz)
                r = np.ones(nz)#-0.3
                zeros_list_r.append(r)
                zeros_list_phi.append(phi)
        
        if set_poles is None:
            poles_list.append(np.concatenate([
                                    np.random.uniform(bounds.poles_r_min,bounds.poles_r_max,nz),
                                    np.random.uniform(bounds.poles_phi_min,bounds.poles_phi_max,nz)
                                    ]))
    if set_poles is None:
        zeros = np.concatenate(zeros_list_r+zeros_list_phi)
        poles = np.concatenate(poles_list)    
        m0 = np.concatenate([zeros,poles])
        return m0
    
    zeros = np.concatenate(zeros_list_r+zeros_list_phi)
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
        bounds.zeros_r_max = 0.99
        bounds.zeros_phi_max = np.pi
        bounds.poles_r_max = 0.99
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
        bounds.zeros_r_max = 3
        bounds.zeros_phi_max = np.pi
        bounds.poles_r_max = 3
        bounds.poles_phi_max = np.pi

        bounds.zeros_r_min = 1/bounds.zeros_r_max
        bounds.zeros_phi_min = 0
        bounds.poles_r_min = 1/bounds.poles_r_max
        bounds.poles_phi_min = 0
        bounds.phase = 'LINEAR'
    else:
        raise ValueError('Unrecognised Type')
    return bounds
    

def sort_bounds(bounds_list,coeffs_per_stage,set_poles):
    bounds_zeros_r = []
    bounds_zeros_phi = []
    for bounds,nz in zip(bounds_list,coeffs_per_stage):
        if bounds.phase == 'MIN':
            bounds_zeros_r += [(bounds.zeros_r_min,bounds.zeros_r_max) for _ in range(nz)]  #Real
            bounds_zeros_phi+=[(bounds.zeros_phi_min,bounds.zeros_phi_max) for _ in range(nz)]  #Imag
        elif bounds.phase =='LINEAR':
            bounds_zeros_r += [(bounds.zeros_r_min,bounds.zeros_r_max) for _ in range(nz)]  #Real
            bounds_zeros_phi+=[(bounds.zeros_phi_min,bounds.zeros_phi_max) for _ in range(nz)]  #Imag
        
        if set_poles is not None:
            continue
        bounds_poles = [(bounds.poles_r_min,bounds.poles_r_max) for _ in range(nz)]  #Real
        bounds_poles+=[(bounds.poles_phi_min,bounds.poles_phi_max) for _ in range(nz)]  #Imag

    ret = bounds_zeros_r+bounds_zeros_phi
    return ret

def get_system(type:str,phases_per_stage:list[str],coeffs_per_stage:list[int]):
    """
    Get the required bounds and settings for the specified parameters. 
    Return the initialiased poles and zeros and the bounds to be given to optimiser
    Args:
        type:   'FIR' or 'IIR',
        phase:   'MIN' or 'LINEAR' or None
    """
    nz = sum(coeffs_per_stage)*2 
    #   Need to claculate the correct size for set_poles
    #   Need a list of bounds namespaces
    set_poles_size = 0
    bounds_list = []
    for phase in phases_per_stage:
        if type=='FIR':
            if phase == 'MIN':
                set_poles_size += nz
            elif phase == 'LINEAR':
                set_poles_size += nz
        elif 'IIR':
            set_poles = None
            bounds = bounds
        else:
            raise ValueError('Unknown Filter Type')
        
        if phase is None :
            bounds_list.append( get_bounds(phase))
        elif phase == 'MIN' and type=='FIR':
            bounds_list.append( get_bounds(phase))
        elif phase == 'LINEAR':
            bounds_list.append( get_bounds(phase))
        else:
            raise ValueError('Unknown Type/Phase pair')
    set_poles = np.zeros(set_poles_size)
    m0 = initialise_model(coeffs_per_stage,bounds_list,set_poles)
    ret_bounds = sort_bounds(bounds_list,coeffs_per_stage,set_poles)
    return m0,ret_bounds,set_poles,nz


def sort_mpost(m_post,nz,set_poles,phases_per_stage,coeffs_per_stage):
    pandz_list = []
    print(m_post.shape)
    print(nz)
    for i,(phase,nc) in enumerate(zip(phases_per_stage,coeffs_per_stage)):
        if phase=='MIN' or phase == None:
            if set_poles is not None:
                poles_final = set_poles[i*nc:(i+1)*nc]*np.exp(1j*set_poles[nz//2 + i*nc:nz//2 + (i+1)*nc]) #+ 1j*set_poles[nz//2:]
            else:
                poles_final = m_post[nz:(3*nz)//2]*np.exp(1j*m_post[(3*nz)//2:])#+1j*m_post[(3*nz)//2:]

            zeros_final = m_post[i*nc:(i+1)*nc]*np.exp(1j*m_post[nz//2 + i*nc:nz//2 + (i+1)*nc]) #+ 1j*m_post[nz//2:nz]


            pandz_list.append(np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])]))
           
        

        elif phase=='LINEAR':
            if set_poles is not None:
                poles_final = set_poles[i*nc:(i+1)*nc]*np.exp(1j*set_poles[nz//2 + i*nc:nz//2 + (i+1)*nc])
            else:
                poles_final = m_post[nz//2:(3*nz)//4]*np.exp(1j*m_post[(3*nz)//4:])

            zeros_final = m_post[i*nc:(i+1)*nc]*np.exp( 1j*m_post[nz//2 + i*nc:nz//2 + (i+1)*nc])

        
            zeros_final = np.concatenate([zeros_final,1/zeros_final])
            if poles_final.shape[0]!=zeros_final.shape[0]:
                to_pad = poles_final.shape[0]-zeros_final.shape[0]
                if to_pad >1:
                    poles_final = poles_final[:poles_final.shape[0]-to_pad]
                elif to_pad <1:
                    poles_final = np.concatenate([poles_final,np.zeros(-to_pad)])

            pandz_list.append(np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),
                                    np.concatenate([zeros_final,np.conj(zeros_final)])]))
            
    pandz = np.vstack(pandz_list)
    return pandz


def main_lbfgs(paths,coeffs_per_stage,phases_per_stage,f_type='FIR',f_phase='MIN',nepochs=100,new_optimise=True,ftol=1e-3,sinc_dec = [256]):
    FIGPATH = 'figures/bfgs'
    X_spectra,Y_spectra,frequencies,pulses = load_datasets(paths,sinc_dec)
    print('Datasets Loaded')

    if new_optimise:
        m0,bounds,set_poles,nz = get_system(f_type,phases_per_stage,coeffs_per_stage)
        print('Got Settings')
        print('Optimsing...')
  
        nz =  sum(coeffs_per_stage)*2

        print(m0)
        print(nz)
     
        res = minimize(
                lambda m:objective_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,phases_per_stage,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles,phase=f_phase),
                x0=m0,
                #jac = lambda m: jac_lbfgs(m,nz,X_spectra,Y_spectra,frequencies,phases_per_stage,scipy=True,coeffs_per_stage=coeffs_per_stage,set_poles = set_poles,phase=f_phase),
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
    
        pandz = sort_mpost(m_post,nz,set_poles,phases_per_stage,coeffs_per_stage)
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
                Path('/run/media/obic/SSD/test/ADC_Filter_4'),
                Path('/run/media/obic/SSD/test/ADC_Filter_5')    
            ]
    #   Number of roots in the top half of the compelx plane
    #   These will by conjugated, so the total order will be twice this number
    #   or in the case of linear phase will be 4 times this number from further symmetries
    #   The numbers need to be divisible by 4 for linear phase.

    #   We looking for a cascade of two linear filters followed by two min phase filters

    #   coeffs per stage is the number of coeffs we seek, so for minphase it'll be 2 times this and 4 times this for linear
    coeffs_per_stage = [128]
    phases_per_stage = ['MIN']
    #   Decimations for the sinc filters
    sinc_dec = None#[2,4,16]
    
    new_optimise = True
    main_lbfgs(paths,coeffs_per_stage,phases_per_stage,new_optimise=new_optimise,ftol=1e-5,sinc_dec = sinc_dec,f_phase='LINEAR',nepochs=100)