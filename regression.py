import numpy as np
import scipy
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path
import tqdm
from numpy.polynomial import Polynomial
from filter_coeffs import *
from scipy.optimize import minimize

def load_pulses(pulses_path):
    pulses = np.loadtxt(pulses_path,delimiter='\t')
    return pulses

def plot_xy(X_spec,Y_spec,frequencies,nplots=4,outfile='data_test.png'):
    n = len(Y_spec)%nplots
    n = (len(Y_spec)-n)//nplots
    Y_spec_plot = Y_spec[::n]
    X_spec_plot = X_spec[::n]
    fig,ax  = plt.subplots(nplots,2,layout='constrained')
    for i in range(nplots):
        ax[i,0].plot(frequencies,X_spec_plot[i])
        ax[i,1].plot(frequencies,Y_spec_plot[i])

    ax[0,0].set_title('Inputs')
    ax[0,1].set_title('Observed')  
    ax[nplots-1,0].set_xlabel('Frequency (Hz)')
    ax[nplots-1,1].set_xlabel('Frequency (Hz)')    
    plt.savefig(outfile,dpi=256)

def cut_timeseries(y,chattr,start_times,length_seconds):
    dataset_start_time = chattr['data_start']
    sample_rate = chattr['sample_rate_hz']
    if dataset_start_time <= 86400:
        dataset_start_time = start_times[0]
    start_samples = ((start_times - dataset_start_time)*sample_rate).astype(np.int64)
    length_samples = (length_seconds*sample_rate).astype(np.int64)
    ret = []
    for start,length in zip(start_samples,length_samples):
        ret.append(y[start:start+length])

    return ret


def data_transform(spectra):
    ret = []
    eps = 1e-5
    for s in spectra:
        ts = (s**2)**0.5
        ts+=eps
        ret.append(np.log10(ts))
    return ret


def load_xy(pulses_path,data_path):

    pulses = load_pulses(pulses_path)
    dataset = nc.Dataset(data_path)
    y = dataset.variables['ch02'][:]
    chattr = dataset.variables['ch02'].__dict__

    X_known_freqs = pulses[:,1]
    x_amplitudes = pulses[:,2]
    x_cycles = pulses[:,-1]
    x_start_times = pulses[:,0]
    #   The length of the observed timeseries
    length_seconds = x_cycles/X_known_freqs
    #   List of cut observed time series
    y_time  = cut_timeseries(y,chattr,x_start_times,length_seconds)

    n = 6000    #   The length of the spectra
    #   Calculate Y spectra
    Y_spectra = [np.fft.rfft(y,n=n) for y in y_time]
    Y_frequencies = np.fft.rfftfreq(n=n,d=1/chattr['sample_rate_hz'])

    target_high = 1500  #The highest frequency we're interested in
    df_y = Y_frequencies[1]-Y_frequencies[0]
    max_f = Y_frequencies.max()
    #We need to pad the frequencies to get to 1500 Hz
    pad_number = int((target_high - max_f)/df_y)
    Y_frequencies = np.concatenate([Y_frequencies,np.linspace(max_f,target_high,pad_number)])
    Y_spectra = [np.concatenate([y,np.zeros(pad_number)]) for y in Y_spectra]
    n = Y_frequencies.shape[0]

    #   Calculate X spectra       

    x_timeseries = []

    for amplitude,frequency,length in zip(x_amplitudes,X_known_freqs,length_seconds):
        t = np.arange(0,length,1/(2*target_high))  
        x_timeseries.append(amplitude*np.sin(2*np.pi*frequency*t))
    #   Bruh
    X_spectra = [np.fft.rfft(x,n=2*n-1) for x in x_timeseries]
    X_frequencies = np.fft.rfftfreq(n=2*n-1,d=1/(2*target_high))
    
    fig,ax = plt.subplots()
    ax.plot(X_frequencies,X_spectra[-1]/X_spectra[-1].max())
    ax.plot(Y_frequencies,Y_spectra[-1]/Y_spectra[-1].max())
    print(max(X_frequencies),max(Y_frequencies))
    plt.savefig('test.png')
    
    plot_xy(X_spectra,Y_spectra,X_frequencies,5,outfile='Pre_transform_data.png')
    Y_spectra = data_transform(Y_spectra)
    plot_xy(X_spectra,Y_spectra,X_frequencies,5,outfile='Post_transform_data.png')
    return X_spectra,Y_spectra,X_frequencies
 

def create_G(poles,zeros,X_spectra,Y_spectra,frequencies):
    """This creates the Jacobian"""
    G = np.zeros(shape=(len(X_spectra),poles.shape[0]+zeros.shape[0]),dtype=np.complex128)
    eps = 1e-12
    epsi = eps*1j
    for m in range(zeros.shape[0]):
        bp = zeros.copy()
        bp[m] +=eps
        bn = zeros.copy()
        bn[m] -=eps
        diff = (g(poles,bp,X_spectra,Y_spectra,frequencies)-g(poles,bn,X_spectra,Y_spectra,frequencies))/eps

        G[:,m] = diff

    for m in range(poles.shape[0]):
        
        ap = poles.copy()
        ap[m] +=eps+epsi
        an = poles.copy()
        an[m] -=eps+epsi
        diff = (g(ap,zeros,X_spectra,Y_spectra,frequencies)-g(an,zeros,X_spectra,Y_spectra,frequencies))/eps

        G[:,zeros.shape[0]+m] = diff
    return G

def L2_norm(d_obs,d):
    return np.sqrt((d_obs-d)@(d_obs-d).T)

def calculate_transfer_function(poles,zeros,omega):
    assert poles.shape[0]==zeros.shape[0]
    num=1
    den = 1
    for ma,mb in zip(poles,zeros):
        num *=(omega-mb)
        den *= (omega-ma)
    return num/den

def g(poles,zeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray,data_only=False):
    """The forward model for a list of spectra and frequencies"""
    ret = []
    #   We only search for poles in the upper left quadrant, and append the remaining co
    zeros = np.concatenate([zeros,np.conj(zeros)])
    poles = np.concatenate([poles,np.conj(poles)])
    #   Do everything in rad/s and we have 
    omega =2*np.pi*frequencies
    for X in X_spectra:
        num=np.ones_like(X,dtype=np.complex128)
        den = np.ones_like(X,dtype=np.complex128)
        for ma,mb in zip(poles,zeros):
            num *=(omega-mb)
            den *= (omega-ma)
        
        ret.append((num/den)*X)
    ret = data_transform(ret)
    if data_only:
        return ret
    ret = [L2_norm(d_obs,d) for d_obs,d in zip(Y_spectra,ret)]
    return np.array(ret)

def hess(G):
    return G.T@G

def jac(G,m,nzeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray):
    zeros = m[:nzeros//2]
    poles = m[nzeros//2:]
    
    return G.T@(-g(poles,zeros,X_spectra,Y_spectra,frequencies))


def model_update(m_last,nzeros,Qm,X_spectra,Y_spectra,frequencies):
    zeros = m_last[:nzeros//2]
    poles = m_last[nzeros//2:]
    G = create_G(poles,zeros,X_spectra,Y_spectra,frequencies)

    H = hess(G)
    s = jac(G,m_last,nzeros,X_spectra,Y_spectra,frequencies)
    m_post  = m_last+ np.linalg.solve(H+Qm,s)
    return m_post

def optimise(nz,X_spectra,Y_spectra,frequencies,nepochs=100,alpha=1e-2):
    """Optimise for the poles and zeros"""

    poles = np.random.uniform(-2000,0,nz//2)+1j*np.random.uniform(0,2000,nz//2)#-np.linspace(0.1,100,nz//2)+np.linspace(0j,100j,nz//2)
    zeros = np.random.uniform(-2000,2000,nz//2)+1j*np.random.uniform(0,2000,nz//2)#np.zeros(nz//2)#np.linspace(-100,100,nz)
    #zeros[:nz//4] = 500*np.pi

    #   Zeros First, Poles second
    m0 = np.concatenate([zeros,poles])
    Qm = np.eye(poles.shape[0]+zeros.shape[0])*alpha
    mi = m0
    losses = []
    for e in tqdm.tqdm(range(nepochs),desc = 'Optimising'):
        mi1 = model_update(mi,nz,Qm,X_spectra,Y_spectra,frequencies)
        loss  = g(mi[nz//2:],mi[:nz//2],X_spectra,Y_spectra,frequencies)
        losses.append(sum(loss**2)**0.5)
        mi = mi1
        # mi[:nz] = mi[:nz].real
        #mi[nz//2:] = -abs(mi[nz//2:].real)+1j*mi[nz//2:].imag
        poles_i = mi[nz//2:]
        mask = poles_i.real > 0
        poles_i[mask] = 0 +1j*poles_i[mask].imag
        mi[nz//2:] = poles_i
        
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
    plt.savefig('Nz.png')

    grad = np.diff(log_d_norms)/np.diff(log_m_norms)
    min_ind = np.argmin(grad)
    nz_opt = (nzs[min_ind+1]-nzs[min_ind])//2
    return int(nz_opt)

def example_filter():
    coeffs = STAGE1_LINEAR
    transfer_function = Polynomial(coeffs,domain=[0,6000])
    transfer_function.roots()
    return 


def objective_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies):
    zeros = m[:nzeros//2] +  m[nzeros//2:nzeros]*1j
    poles = m[nzeros:(3*nzeros)//2]+m[(3*nzeros)//2:]*1j
    l = -g(poles,zeros,X_spectra,Y_spectra,frequencies)
    return np.sqrt(l.__abs__()@l.__abs__().T)

def jac_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies):
    zeros = m[:nzeros//2] +  m[nzeros//2:nzeros]*1j
    poles = m[nzeros:(3*nzeros)//2]+m[(3*nzeros)//2:]*1j
    G = create_G(poles,zeros,X_spectra,Y_spectra,frequencies)
    G = np.column_stack([G[:,:nzeros//2].real,G[:,:nzeros//2].imag,G[:,nzeros//2:].real,G[:,nzeros//2:].imag])

    return G@np.ones(G.shape[1])

def callback_lbfgs(intermediate_result):
    print(intermediate_result)
    print(f'Iteration Loss: {intermediate_result.fun}')
    return

def main_lbfgs():
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_0')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra,Y_spectra,frequencies  = load_xy(pulses_path,nc_path)
    alpha = 1
    nepochs = 100
    new_optimise = True
    if new_optimise:

        nz = 100 
        poles = np.concatenate([np.random.uniform(-2000,0,nz//2),np.random.uniform(0,2000,nz//2)])#-np.linspace(0.1,100,nz//2)+np.linspace(0j,100j,nz//2)
        zeros = np.concatenate([np.random.uniform(-2000,2000,nz//2),np.random.uniform(0,2000,nz//2)])#np.zeros(nz//2)#np.linspace(-100,100,nz)
        

        m0 = np.concatenate([zeros,poles])
        bounds_poles = [(None,0) for _ in range(nz//2)]
        bounds_poles.append([(0,None) for _ in range(nz//2)])

        bounds_zeros = [(None,None) for _ in range(nz//2)]
        bounds_zeros.append([(0,None) for _ in range(nz//2)])

        bounds = bounds_zeros.append(bounds_poles)
        G0 = create_G(m0[nz:(3*nz)//2]+m0[(3*nz)//2:]*1j,m0[:nz//2] +  m0[nz//2:nz]*1j,X_spectra,Y_spectra,frequencies)
        G0 = np.column_stack([G0[:,:nz//2].real,G0[:,:nz//2].imag,G0[:,nz//2:].real,G0[:,nz//2:].imag])

        H0_inv = np.linalg.inv(G0.T@G0 + alpha*np.eye(G0.shape[1]))
        fig,ax = plt.subplots(2)
        ax[0].imshow(np.log10(H0_inv.real))
        ax[1].imshow(np.triu(H0_inv.real)==np.tril(H0_inv.real).T)
        plt.savefig('test.png')


        res = minimize(
                lambda m:objective_lbfgs(m,nz,X_spectra,Y_spectra,frequencies),
                x0=m0,
                jac = lambda m: jac_lbfgs(m,nz,X_spectra,Y_spectra,frequencies),
                method='BFGS',  
                bounds=bounds,
                callback=callback_lbfgs,
                options={'disp':True,
                         'hess_inv0':H0_inv
                         }      
                )
        m_post = res.x

        poles_final = m_post[nz:(3*nz)//2]+1j*m_post[(3*nz)//2:]
        zeros_final = m_post[:nz//2] + 1j*m_post[nz//2:nz]
        pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])])
        np.save('PolesandZeros.npy',pandz)
        C_post = calculate_C_posterior(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,alpha)
        np.save('PosteriorCovariance.npy',C_post)
        data_reconst = g(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,True)

        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
    else:
        C_post = np.load('PosteriorCovariance.npy')
        pandz =  np.load('PolesandZeros.npy')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
        nz = zeros_final.shape[0]
        data_reconst = g(poles_final[nz//2:],zeros_final[:nz//2],X_spectra,Y_spectra,frequencies,True)

    fig,ax = plt.subplots(2,layout='constrained')
    H = calculate_transfer_function(poles_final,zeros_final,2*np.pi*frequencies)
    print(H)
    ax[0].plot(frequencies,H.real/(2*np.pi))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(r'$\mathfrak{R}$')
    ax[1].set_ylabel(r'$\mathfrak{I}$')
    ax[1].plot(frequencies,H.imag)
    plt.savefig('Transfer_function.png')
    plt.close()

    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real,poles_final.imag,'x',label='Poles')
    ax.plot(zeros_final.real,zeros_final.imag,'o',label='Zeros')
    ax.set_ylabel(r'$\mathfrak{Im} (rad/s)$')
    ax.set_xlabel(r'$\mathfrak{Re} (rad/s)$')
    ax.grid()
    plt.savefig('PoleandZeros.png')
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real/(2*np.pi),poles_final.imag/(2*np.pi),'x',label='Poles')
    ax.plot(zeros_final.real/(2*np.pi),zeros_final.imag/(2*np.pi),'o',label='Zeros')
    ax.set_ylabel(r'$\mathfrak{Im} (Hz)$')
    ax.set_xlabel(r'$\mathfrak{Re} (Hz)$')
    ax.grid()
    plt.savefig('PoleandZerosHz.png')


    fig,(ax,ax1) = plt.subplots(2,layout='constrained')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$||\Delta d||_{2}^{2}$')
    for y,d in zip(Y_spectra,data_reconst):
        ax1.plot(frequencies,y,'r')
        ax1.plot(frequencies,d,'k')
    fig.savefig('losses.png')
    plt.close()
    fig,ax = plt.subplots(layout='constrained')
    im = ax.imshow(C_post.real)
    plt.colorbar(im,ax=ax,label='Posterior Variance')
    plt.savefig('PosteriorCovar.png',dpi=256)
    return

def main():
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_0')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra,Y_spectra,frequencies  = load_xy(pulses_path,nc_path)
    alpha = .5
    nepochs = 100
    new_optimise = True
    if new_optimise:
        nzs = np.linspace(10,100,10,dtype=np.int64)
        #nz_optimum = determine_nz(nzs,X_spectra,Y_spectra,frequencies,100)
        nz = 50 #nz_optimum
        m_post,losses = optimise(nz,X_spectra,Y_spectra,frequencies,nepochs,alpha)
        poles_final = m_post[nz//2:]
        zeros_final = m_post[:nz//2]
        pandz = np.column_stack([np.concatenate([poles_final,np.conj(poles_final)]),np.concatenate([zeros_final,np.conj(zeros_final)])])
        np.save('PolesandZeros.npy',pandz)
        C_post = calculate_C_posterior(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,alpha)
        np.save('PosteriorCovariance.npy',C_post)
        data_reconst = g(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,True)

        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
    else:
        C_post = np.load('PosteriorCovariance.npy')
        pandz =  np.load('PolesandZeros.npy')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
        nz = zeros_final.shape[0]
        data_reconst = g(poles_final[nz//2:],zeros_final[:nz//2],X_spectra,Y_spectra,frequencies,True)

    fig,ax = plt.subplots(2,layout='constrained')
    H = calculate_transfer_function(poles_final,zeros_final,2*np.pi*frequencies)
    print(H)
    ax[0].plot(frequencies,H.real/(2*np.pi))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(r'$\mathfrak{R}$')
    ax[1].set_ylabel(r'$\mathfrak{I}$')
    ax[1].plot(frequencies,H.imag)
    ax[0].axvline(x=250)
    ax[1].axvline(x=250)

    ax[0].loglog()
    ax[1].loglog()
    
    plt.savefig('Transfer_function.png')
    plt.close()

    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real,poles_final.imag,'x',label='Poles')
    ax.plot(zeros_final.real,zeros_final.imag,'o',label='Zeros')
    ax.set_ylabel(r'$\mathfrak{Im} (rad/s)$')
    ax.set_xlabel(r'$\mathfrak{Re} (rad/s)$')
    ax.grid()
    plt.savefig('PoleandZeros.png')
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real/(2*np.pi),poles_final.imag/(2*np.pi),'x',label='Poles')
    ax.plot(zeros_final.real/(2*np.pi),zeros_final.imag/(2*np.pi),'o',label='Zeros')
    ax.set_ylabel(r'$\mathfrak{Im} (Hz)$')
    ax.set_xlabel(r'$\mathfrak{Re} (Hz)$')
    ax.grid()
    plt.savefig('PoleandZerosHz.png')


    fig,(ax,ax1) = plt.subplots(2,layout='constrained')
    if new_optimise:
        ax.plot(losses)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$||\Delta d||_{2}^{2}$')
    for y,d in zip(Y_spectra,data_reconst):
        ax1.plot(frequencies,y,'r')
        ax1.plot(frequencies,d,'k')
    fig.savefig('losses.png')
    plt.close()
    fig,ax = plt.subplots(layout='constrained')
    im = ax.imshow(C_post.real)
    plt.colorbar(im,ax=ax,label='Posterior Variance')
    plt.savefig('PosteriorCovar.png',dpi=256)

if __name__ == '__main__':
    main()