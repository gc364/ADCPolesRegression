import numpy as np
import scipy
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path
import tqdm
from numpy.polynomial import Polynomial
from filter_coeffs import *
from scipy.optimize import minimize
import numpy.polynomial.polynomial as poly
from scipy.signal import freqs_zpk
from datetime import datetime

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
        ax[i,0].plot(frequencies,X_spec_plot[i].real)
        ax[i,1].plot(frequencies,Y_spec_plot[i].real)

    ax[0,0].set_title('Inputs')
    ax[0,1].set_title('Observed')  
    ax[nplots-1,0].set_xlabel('Frequency (Hz)')
    ax[nplots-1,1].set_xlabel('Frequency (Hz)')    
    plt.savefig(outfile,dpi=256)

def to_coefficents(poles,zeros):
    """Convert poles and zeros to numerator and denominator coefficients"""
    num = poly.polyfromroots(zeros)
    den = poly.polyfromroots(poles)

    return num,den

def scipy_frequency_response(poles,zeros,frequencies=None,gain=1):
    """Compute the frequency response using scipy"""
    if type(frequencies) == None:
        w,H = freqs_zpk(zeros,poles,gain,worN=1500,fs=2*np.pi*500)
        
        return w,H
    w,H = freqs_zpk(zeros,poles,gain,frequencies)
  
    return w,H


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
    
    # fig,ax = plt.subplots()
    # ax.plot(X_frequencies,X_spectra[-1]/X_spectra[-1].max())
    # ax.plot(Y_frequencies,Y_spectra[-1]/Y_spectra[-1].max())
    # print(max(X_frequencies),max(Y_frequencies))
    # plt.savefig('test.png')
    
    plot_xy(X_spectra,Y_spectra,X_frequencies,5,outfile='Pre_transform_data.png')
    Y_spectra = data_transform(Y_spectra)
    plot_xy(X_spectra,Y_spectra,X_frequencies,5,outfile='Post_transform_data.png')
    return X_spectra,Y_spectra,X_frequencies


def calculate_transfer_function(poles,zeros,omega):
    assert poles.shape[0]==zeros.shape[0]
    num=1
    den = 1
    for ma,mb in zip(poles,zeros):
        num *=(omega-mb)
        den *= (omega-ma)
    return num/den

def L2_norm(d_obs,d):
    return np.sqrt((d_obs-d)@(d_obs-d).T)


def g(poles,zeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray,data_only=False):
    """The forward model for a list of spectra and frequencies"""
    ret = []
    #   We only search for poles in the upper left quadrant, and append the remaining co
    zeros = np.concatenate([zeros,np.conj(zeros)])
    poles = np.concatenate([poles,np.conj(poles)])
    #   Do everything in rad/s and we have 
    omega =2*np.pi*frequencies*1j
    for X in X_spectra:
        num=np.ones_like(X,dtype=np.complex128)
        den = np.ones_like(X,dtype=np.complex128)
        for ma,mb in zip(poles,zeros):
            num *=(omega-mb)
            den *= (omega-ma)
        spec = (num/den)*X
        #   zero anything past nyqvist
        ind = np.argmin(abs(frequencies-250))
        spec[ind:] = 0
        ret.append(spec)
    ret = data_transform(ret)
    if data_only:
        return ret
    ret = [L2_norm(d_obs,d) for d_obs,d in zip(Y_spectra,ret)]
    return np.array(ret)

def g_scipy(poles,zeros,X_spectra:list[np.ndarray],Y_spectra:list[np.ndarray],frequencies:np.ndarray,data_only=False):
    """The forward model for a list of spectra and frequencies"""
    ret = []
    #   We only search for poles in the upper left quadrant, and append the remaining co
    zeros = np.concatenate([zeros,np.conj(zeros)])
    poles = np.concatenate([poles,np.conj(poles)])
    w,h  = scipy_frequency_response(poles,zeros,2*np.pi*frequencies)
 
    ret = []
    for x in X_spectra:
        spec = h*x
        #   zero anything past nyqvist
        ind = np.argmin(abs(frequencies-250))
        spec[ind:] = 0
        ret.append(spec)
    ret = data_transform(ret)
    if data_only:
        return ret
    ret = [L2_norm(d_obs,d) for d_obs,d in zip(Y_spectra,ret)]
    return np.array(ret)


def compute_group_delay(phase,frequencies):
    
    return np.diff(phase)/np.diff(frequencies)


NICE_FIGURES = Path('./figures/nice_figures')
def create_nice_figures(poles,zeros,X_spectra,Y_spectra,nc_path,pulses,data_frequencies):
    NICE_FIGURES.mkdir(exist_ok=True)
    mask =poles.real > 0
    poles[mask] = -abs(poles[mask].real)+1j*poles[mask].imag
    b,a = to_coefficents(poles,zeros)

    b_norm,a_norm = scipy.signal.normalize(b,a)
  
    poles_norm = poly.polyroots(a_norm)
    zeros_norm = poly.polyroots(b_norm)

    frequencies_hz = np.logspace(-3,3,5000)
    frequencies_rad = frequencies_hz*2*np.pi
    _,H = scipy_frequency_response(poles,zeros,frequencies_rad)
    phase = np.angle(H)
    fig,ax  = plt.subplots(2,layout='constrained')
    ax[0].plot(frequencies_rad,20*np.log10(H))
    ax[0].semilogx()
    ax[0].set_xlabel(r'$\omega$ (rad/s)')
    ax[0].set_ylabel(r'Response (dB)')
    ax[1].plot(frequencies_rad,phase)
    ax[1].semilogx()
    ax[1].set_xlabel(r'$\omega$ (rad/s)')
    ax[1].set_ylabel(r'Phase (rad)')
    plt.savefig(NICE_FIGURES.joinpath('frequency_response.png'),dpi=256)


    fig,ax  = plt.subplots(2,layout='constrained')
    ax[0].plot(frequencies_hz,20*np.log10(H))
    ax[0].semilogx()
    ax[0].set_xlabel(r'$f$ (Hz)')
    ax[0].set_ylabel(r'Response (dB)')
    ax[1].plot(frequencies_hz,phase)
    ax[1].semilogx()
    ax[1].set_xlabel(r'$f$ (Hz)')
    ax[1].set_ylabel(r'Phase (rad)')
    plt.savefig(NICE_FIGURES.joinpath('frequency_response_hz.png'),dpi=256)

  
    times = np.linspace(0,frequencies_hz.shape[0],frequencies_hz.shape[0])
    t,impulse = scipy.signal.impulse((b_norm,a_norm),T=times)
    fig,(ax,ax1) = plt.subplots(2,layout='constrained')
    ax.plot(t,impulse)
    ax.set_xlabel('samples')
    ax.set_ylabel('Impulse Response')

    t,step = scipy.signal.step((b_norm,a_norm),T=times)

    ax1.plot(t,step)
    ax1.set_xlabel('samples')
    ax1.set_ylabel('Step Response')
    plt.savefig(NICE_FIGURES.joinpath('impulse_step_response.png'),dpi=256)


    fig,ax = plt.subplots(2,layout='constrained')
    b_cumulative  = np.cumsum(b_norm.__abs__())
    b_cumulative *= 1/b_cumulative.max()
    ax[0].plot(np.abs(b_norm))
    ax[0].semilogy()
    
    a_cumulative  = np.cumsum(a_norm.__abs__())
    a_cumulative *= 1/a_cumulative.max()
    ax[1].plot(np.abs(b_norm))
    ax[1].semilogy()

    ax[0].set_ylabel('Numerator Coefficents')
    ax[1].set_ylabel('Denomimator Coefficents')     
    plt.savefig(NICE_FIGURES.joinpath('coefficient_spectra.png')  )


    fig,ax = plt.subplots(layout='constrained')
    _,group_delay = scipy.signal.group_delay((b_norm,a_norm),w=frequencies_rad,fs=2*np.pi*500)
    group_delay = compute_group_delay(phase,frequencies_rad)
    inds = group_delay <=0
    print(group_delay)
    ax.plot(frequencies_hz[1:][inds],group_delay[inds])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (s/rad)')
    plt.savefig(NICE_FIGURES.joinpath('group_delay.png'),dpi=256)

    #   THD at 31.25 Hz signal
    #   Put a 31.25 Hz signal through the filter then look at the calculate the fundamen
    thds = []
    f_tests = [31.25]#np.logspace(-2,2,20)
    for f in f_tests:
        t = np.arange(0,1000,1/500)
        x = np.sin(2*np.pi*f*t)
        X = np.fft.rfft(x)
        sort_ind = np.argsort(X.__abs__())
        harmonics = X[sort_ind][-6:]   #   the top feq peaks
       
        thd = np.sum(harmonics[:-1].__abs__())/harmonics[-1].__abs__()
        thds.append(20*np.log10(thd))
        
    fig,ax = plt.subplots()
    ax.plot(f_tests,thds,'r*')
    ax.set_ylabel('THD (dB)')
    ax.set_xlabel('Frequency (Hz)')
    plt.savefig(NICE_FIGURES.joinpath('THD_synthetic.png'))


    input_freqs = pulses[:,1]
    sorted_freq_ind = np.argsort(input_freqs)
    

    thds = []

    for f,X in zip(input_freqs,X_spectra):

        nyq_ind = np.argmin(abs(data_frequencies-250))

        X[nyq_ind:] = 0

        sort_ind = np.argsort(X.__abs__())
        sort_ind,_ = scipy.signal.find_peaks(X)
       
        harmonics = X[sort_ind] 
        harmonics =np.sort( harmonics[-32:].__abs__() )
        thd = np.sum(harmonics[:-1].__abs__())/harmonics[-1].__abs__()
        thd = thd/np.sqrt(1+thd**2)

        thds.append(20*np.log10(thd))
        
    fig,ax = plt.subplots()
    ax.plot(input_freqs[sorted_freq_ind],np.array(thds)[sorted_freq_ind])
    ax.set_ylabel('THD (dB)')
    ax.set_xlabel('Input Freq (Hz)')
    plt.savefig(NICE_FIGURES.joinpath('THD_observed_X.png'))
    thds = []
    for X in Y_spectra:
        X = 10**X
        sort_ind,_ = scipy.signal.find_peaks(X)
       
        harmonics = X[sort_ind] 
        harmonics =np.sort( harmonics[-32:].__abs__() )
        thd = np.sum(harmonics[:-1].__abs__())/harmonics[-1].__abs__()
        thd = thd/np.sqrt(1+thd**2)
        thds.append(20*np.log10(thd))
        
    fig,ax = plt.subplots()

    ax.plot(input_freqs[sorted_freq_ind],np.array(thds)[sorted_freq_ind])
    ax.set_ylabel('THD (dB)')
    ax.set_xlabel('Input Freq (Hz)')
    plt.savefig(NICE_FIGURES.joinpath('THD_observed_Y.png'))

    #   Circular poles and zeros
    print(poles.max())
    print(zeros.max())
    print(poles.min())
    print(zeros.min())
    zeros_arg = np.abs(zeros)
    poles_arg = np.abs(poles)
    zeros_c = np.exp(1j*np.angle(zeros))
    poles_c = np.exp(1j*np.angle(poles))

    

    fig,ax = plt.subplots()
    ax.plot(zeros_c.real,zeros_c.imag,'o')
    ax.plot(poles_c.real,poles_c.imag,'x')
    ax.grid()
    f_obs = 31.25
    f_obs_c = np.exp(1j*f_obs*2*np.pi)
    ax.plot(f_obs_c.real,f_obs_c.imag,'*')
    plt.savefig(NICE_FIGURES.joinpath('unit_poles.png'),dpi=256)



    #   SNR 
    #   We need to look for the level of noise with no signal
    #   migth be best to just read somthing at 30s past the minute in the nc files
    #   Then fft and do 10log(P_sig/P_noise)
  
    dataset = nc.Dataset(nc_path,'r')
  
    var = dataset.variables['ch02']
    sample_rate = var.__dict__['sample_rate_hz']
    start_time =  var.__dict__['data_start']
    end_time =  var.__dict__['data_end']

    noise_start =datetime.fromtimestamp(start_time + ((end_time-start_time)*3/4))
    noise_start = datetime(noise_start.year,
                           noise_start.month,
                           noise_start.day,
                           noise_start.hour,
                           noise_start.minute
                           ).timestamp()+30
    noise_end = noise_start+20

    start_ind = int((noise_start-start_time)*sample_rate)
    end_ind = int((noise_end-start_time)*sample_rate)
    noise = var[start_ind:end_ind]
    noise_prime = np.fft.rfft(noise,n = 6000)
    noise_freqs = np.fft.rfftfreq(n=6000,d=1/sample_rate)

    target_high = 1500  
    df_y = noise_freqs[1]-noise_freqs[0]
    max_f = noise_freqs.max()
    pad_number = int((target_high - max_f)/df_y)
    noise_freqs = np.concatenate([noise_freqs,np.linspace(max_f,target_high,pad_number)])
    noise_prime = np.concatenate([noise_prime,np.zeros(pad_number)]) 
    noise_prime = data_transform(noise_prime)
    


    snr = 10*(Y_spectra[0]-noise_prime)
    fig,ax = plt.subplots(layout='constrained')
    ax.plot(noise_freqs,snr)

    ax.set_ylabel('SNR (dB)')
    ax.set_xlabel('Frequency (Hz)')
    plt.savefig(f'{NICE_FIGURES}/SNR.png',dpi=256)

    #   Look at multiples of the input signal and the sampling rate for harmonic imperfections
    
    fig,ax = plt.subplots()
    ind = 30
    ax.plot(noise_freqs,Y_spectra[ind])
    for i in range(1,10):
        ax.axvline(x = input_freqs[ind]*i)
    ax.set_xlim(0,250)
    plt.savefig(f'{NICE_FIGURES}/SFDR.png')

  
    return