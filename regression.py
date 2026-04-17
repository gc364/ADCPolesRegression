import numpy as np
import scipy
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path

"""TODO:    This need to be essentially FWI but for the spectra, so we need to implement a sense of spectra difference"""




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

    target_high = 1500
    df_y = Y_frequencies[1]-Y_frequencies[0]
    max_f = Y_frequencies.max()
    #We need to pad the frequencies to get to 1500 Hz
    pad_number = int((target_high - max_f)/df_y)
    Y_frequencies = np.concatenate([Y_frequencies,np.linspace(max_f,target_high,pad_number)])
    Y_spectra = [np.concatenate([y,np.zeros(pad_number)]) for y in Y_spectra]
    n = Y_frequencies.shape[0]

    #   Calculate X spectra       

    #   This creates a spike in frequency domain, I'm no sure it works properly
    # X_frequencies = Y_frequencies
    # X_spectra = [np.zeros(X_frequencies.shape[0]) for _ in range(len(Y_spectra))]
    # #   Need the indices
    # indices = []
    # for target in X_known_freqs:
    #     indices.append(X_frequencies.round()==round(target))
    # for ind,spec in zip(indices,range(len(X_spectra))):
    #     X_spectra[spec][ind] = 1
    # fig,ax = plt.subplots()
    # ax.plot(X_frequencies,X_spectra[4])
    # plt.savefig('test.png')   
    # fig,ax = plt.subplots()
    # ax.plot(np.fft.irfft(X_spectra[0]))
    # plt.savefig('test.png')

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

    for m in range(zeros.shape[0]):
        bp = zeros.copy()
        bp[m] +=eps
        bn = zeros.copy()
        bn[m] -=eps
        diff = (g(poles,bp,X_spectra,Y_spectra,frequencies)-g(poles,bn,X_spectra,Y_spectra,frequencies))/eps

        G[:,m] = diff

    for m in range(poles.shape[0]):
        
        ap = poles.copy()
        ap[m] +=eps
        an = poles.copy()
        an[m] -=eps
        diff = (g(ap,zeros,X_spectra,Y_spectra,frequencies)-g(an,zeros,X_spectra,Y_spectra,frequencies))/eps

        G[:,zeros.shape[0]+m] = diff

    return G

def L2_norm(d_obs,d):
    return np.sqrt((d_obs-d)@(d_obs-d).T)

def g(poles,zeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray,data_only=False):
    """The forward model for a list of spectra and frequencies"""
    ret = []
    for X in X_spectra:
        num=np.ones_like(X,dtype=np.complex128)
        den = np.ones_like(X,dtype=np.complex128)
        for ma,mb in zip(poles,zeros):
            num *=(frequencies-mb)
            den *= (frequencies-ma)
        
        ret.append((num/den)*X)
    ret = data_transform(ret)
    if data_only:
        return ret
    ret = [L2_norm(d_obs,d) for d_obs,d in zip(Y_spectra,ret)]
    return np.array(ret)

def hess(m,nzeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray):
    zeros = m[:nzeros]
    poles = m[nzeros:]
    

    G = create_G(poles,zeros,X_spectra,Y_spectra,frequencies)

    return G.T@G

def jac(m,nzeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray):
    zeros = m[:nzeros]
    poles = m[nzeros:]
    

    G = create_G(poles,zeros,X_spectra,Y_spectra,frequencies)

    return G.T@(-g(poles,zeros,X_spectra,Y_spectra,frequencies))

def model_update(m_last,nzeros,Qm,X_spectra,Y_spectra,frequencies):
    H = hess(m_last,nzeros,X_spectra,Y_spectra,frequencies)
    s = jac(m_last,nzeros,X_spectra,Y_spectra,frequencies)
    m_post  = m_last+ np.linalg.solve(H+Qm,s)
    return m_post

def main():
   
    workdir = Path('/run/media/obic/SSD/test/ADC_Filter_0')
    nc_path = list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]
    pulses_path = workdir.joinpath('processed/pulses.txt')
    X_spectra,Y_spectra,frequencies  = load_xy(pulses_path,nc_path)
    poles = np.array([250+2j,250-2j])
    zeros = np.array([-250,250])
    nz = zeros.shape[0]

    #   Zeros First, Poles second
    m0 = np.concatenate([zeros,poles])


    Qm = np.eye(poles.shape[0]+zeros.shape[0])*100
    nepochs = 100
    mi = m0
    losses = []
    for e in range(nepochs):
        mi1 = model_update(mi,nz,Qm,X_spectra,Y_spectra,frequencies)
        loss  = g(mi[nz:],mi[:nz].real,X_spectra,Y_spectra,frequencies)
        losses.append(sum(loss**2)**0.5)
        mi = mi1
        mi[:nz] = mi[:nz].real
    
    m_post = mi
    print(m_post)
    fig,ax = plt.subplots()
    ax.plot(losses)
    plt.savefig('losses.png')

main()