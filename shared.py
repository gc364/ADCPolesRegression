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
from scipy.signal import freqs_zpk,freqz_zpk
from datetime import datetime


def sinc_filter_coeffs(N: int) -> np.ndarray:
    """
    Return the impulse response of a 5th-order sinc (CIC) filter with
    decimation ratio N by convolving the rectangular window 5 times.

    Parameters
    ----------
    N : int
        Decimation ratio (from SINC_DECIMATION table).

    Returns
    -------
    h : np.ndarray
        FIR coefficients (length = 5*(N-1)+1), normalised to unity DC gain.
    """
    rect = np.ones(N)
    h = rect.copy()
    for _ in range(4):  # convolve 5 times total
        h = np.convolve(h, rect)
    h = h / h.sum()  # normalise so DC gain = 1
    return h



def sinc_filter_response(N: int,frequencies,f_mod: float = F_MOD_HIGH, n_points: int = 8192):
    """
    Frequency response of the sinc filter.

    Returns
    -------
    freqs : np.ndarray   Frequency axis (Hz)
    H  : np.ndarray   Magnitude response (not dB)
    """
    h = sinc_filter_coeffs(N)
    w, H = scipy.signal.freqz(h, worN=frequencies, fs=f_mod)
    #H_dB = 20 * np.log10(np.abs(H) + 1e-300)
    return w, H


def apply_sinc_filter(X: np.ndarray, N: int,frequencies) -> np.ndarray:
    """
    Apply the sinc filter 

    Parameters
    ----------
    x : np.ndarray   Input spectra
    N : int          Decimation ratio.

    Returns
    -------
    y : np.ndarray   Filtered and decimated output.
    """
    w,h = sinc_filter_response(N,frequencies)
    y = X*h
    return y

def create_fir_filter_PZs():
    """This returns the zeros and poles for the FIR Filter as a good start point for the optimisation"""
    h1 = STAGE1_LINEAR  # ×2 decimation
    h2 = STAGE2_LINEAR  # ×2 decimation
    h3 = np.array(STAGE3_MINPHASE_RAW)/ SCALE34  # ×4 decimation
    h4 = np.array(STAGE4_MINPHASE_RAW)/ SCALE34  # ×2 decimation



    # Overall FIR decimation = 2*2*4*2 = 32
    # Build the equivalent FIR at the sinc output rate by upsampling each
    # subsequent stage by the cumulative decimation before it.
    def upsample_and_pad(h, factor):
        """Insert (factor-1) zeros between every tap."""
        if factor == 1:
            return h
        out = np.zeros(len(h) * factor - (factor - 1))
        out[::factor] = h
        return out

    # Cascade by convolving upsampled versions
    H_cascade = h1.copy()
    H_cascade = np.convolve(H_cascade, upsample_and_pad(h2, 2))
    H_cascade = np.convolve(H_cascade, upsample_and_pad(h3, 4))
    H_cascade = np.convolve(H_cascade, upsample_and_pad(h4, 16))


    numerator1 = Polynomial(h1)
    zeros1=numerator1.roots()
    numerator2 = Polynomial(h2)
    zeros2=numerator2.roots()
    numerator3 = Polynomial(h3)
    zeros3=numerator3.roots()
    numerator4 = Polynomial(h4)
    zeros4=numerator4.roots()
  




    numerator = Polynomial(H_cascade)
    denominator = Polynomial(np.ones_like(H_cascade))

    zeros = numerator.roots()
    print(zeros)
    poles = denominator.roots()
    print(poles)
    # Frequency response at the sinc output rate
    w, H = scipy.signal.freqz(H_cascade, worN=250, fs=500,whole=True)
    fig,ax = plt.subplots()
    ax.plot(w,H)
    ax.set_ylabel('Transfer function')
    plt.savefig('figures/fir_old.png')

    fig,ax = plt.subplots(5,layout='constrained',figsize=(8.2,11.7))
    print(f'Number of Zeros: {zeros.shape}')
    print(f'Number of Poles: {poles.shape}')
    ax[0].plot(zeros.real,zeros.imag,'o')
    ax[0].plot(poles.real,poles.imag,'x',markersize=4)
    ax[0].grid()
    ax[0].set_ylabel(r'$\mathfrak{Im}$')
    ax[4].set_xlabel(r'$\mathfrak{Re}$')
    ax[0].set_title('Full Cascade')
    ax[0].set_aspect('equal')
    
    ax[1].set_title('Linear')
    
    
    ax[1].plot(zeros1.real,zeros1.imag,'o')
    ax[1].grid()
    ax[1].set_ylabel(r'$\mathfrak{Im}$')
    #ax[3].plot(poles.real,poles.imag,'x',markersize=4)
    ax[1].set_aspect('equal')
    
    ax[2].plot(zeros2.real,zeros2.imag,'o')
    ax[2].grid()
    ax[2].set_ylabel(r'$\mathfrak{Im}$')
    ax[2].set_aspect('equal')

    ax[3].set_title('Min-phase')
    ax[3].plot(zeros3.real,zeros3.imag,'o')
    ax[3].grid()
    ax[3].set_ylabel(r'$\mathfrak{Im}$')
    ax[3].set_aspect('equal')

    ax[4].plot(zeros4.real,zeros4.imag,'o')
    ax[4].grid()
    ax[4].set_ylabel(r'$\mathfrak{Im}$')
    ax[4].set_aspect('equal')
    #ax[4].plot(poles.real,poles.imag,'x',markersize=4)



    plt.savefig('figures/fi_old_pandz.png',dpi=256)

    return zeros,poles

create_fir_filter_PZs()
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

def to_coefficents(poles,zeros,coeffs_per_stage=None):
    """Convert poles and zeros to numerator and denominator coefficients"""

    if coeffs_per_stage is None:
        num = poly.polyfromroots(zeros)
        den = poly.polyfromroots(poles)

        return num,den
    



def scipy_frequency_response(poles,zeros,frequencies=None,gain=1):
    """Compute the frequency response using scipy"""
  
    if type(frequencies) == None:
        w,H = freqz_zpk(zeros,poles,gain,worN=1500,fs=2*np.pi*500)
        
        return w,H
    w,H = freqz_zpk(zeros,poles,gain,frequencies)
  
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
        ts = abs(s)
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
    X_frequencies[0]+=1e-5
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

def apply_stages(poles,zeros,X_spectra,Y_spectra,frequencies,coeffs_per_stage,data_only=False,transfer_function=False):
    stages = [] #   The poles and zeros for each stage
    running = 0
    
    for nc in coeffs_per_stage:
        stages.append((poles[running:running+nc],zeros[running:running+nc]))
       
        running+=nc
    X_i = X_spectra
    for i,stage in enumerate(stages):
      
        if i<len(stages)-1:
            X_i = g_scipy(stage[0],stage[1],X_i,Y_spectra,frequencies,True)
            if transfer_function:
                X_i = [X_i[i]-np.log10(X_spectra[i].__abs__()+1e-5) for i in range(len(X_spectra))]
        else:
            l = g_scipy(stage[0],stage[1],X_i,Y_spectra,frequencies,False)
            X_fin = g_scipy(stage[0],stage[1],X_i,Y_spectra,frequencies,True)
            if transfer_function:
                X_fin = [X_fin[i]-np.log10(X_spectra[i].__abs__()+1e-5) for i in range(len(X_spectra))]
    if data_only:
        return X_fin
    return l
def compute_group_delay(phase,frequencies):
    return -np.diff(phase)/np.diff(frequencies)




def compute_thd(X_spectra,Y_spectra,input_freqs,data_frequencies,num_harmonics=5):

    sorted_freq_ind = np.argsort(input_freqs)

    #   THD of the input signal
    thds = []
    for f,X in zip(input_freqs,X_spectra):
        nyq_ind = np.argmin(abs(data_frequencies-250))
        X[nyq_ind:] = 0
        sort_ind = []
        for i in range(num_harmonics):
            if i== 0 :
                sort_ind.append(np.argmin(abs(data_frequencies-f)))
            elif i>0:
                sort_ind.append(np.argmin(abs(data_frequencies-f*(i))))
        harmonics = X[sort_ind]
        thd = np.sum(harmonics[1:].__abs__())/harmonics[0].__abs__()

        thds.append(thd)
    
    #   In decimal
    thd_in = np.array(thds)[sorted_freq_ind]
    
  

    #   THD of the output signal
    thds = []
    for f,X in zip(input_freqs,Y_spectra):
        X = 10**X
        nyq_ind = np.argmin(abs(data_frequencies-250))
        X[nyq_ind:] = 0
        sort_ind,_ = scipy.signal.find_peaks(X)
        sort_ind = []
        for i in range(num_harmonics):
            if i== 0 :
                sort_ind.append(np.argmin(abs(data_frequencies-f)))
            elif i>0:
                sort_ind.append(np.argmin(abs(data_frequencies-f*(i))))
        harmonics = X[sort_ind]
        thd = np.sum(harmonics[1:].__abs__())/harmonics[0].__abs__()

        thds.append(thd)
        
    #   in deicmal
    thd_out = np.array(thds)[sorted_freq_ind]

    #   Work out the distortion between in and out signal and convert to decibels
    ret = 20*np.log10((thd_out-thd_in))

    return ret, input_freqs[sorted_freq_ind]

NICE_FIGURES = Path('./figures/nice_figures')
def create_nice_figures(poles,zeros,X_spectra,Y_spectra,workdir,pulses,data_frequencies,coeffs_per_stage):
    NICE_FIGURES.mkdir(exist_ok=True)
    
    nc_path =  list(workdir.joinpath('processed/netcdf').glob('*.nc'))[0]

    b,a = to_coefficents(poles,zeros)
    print(a.shape)
    print(b.shape)
    if a.shape[0] < b.shape[0]:
        a= np.repeat(a,(b.shape[0]//a.shape[0]))
    print(a.shape)
    print(b.shape)
    #b_norm,a_norm = scipy.signal.normalize(b,a)
    a_norm = a
    b_norm=b
    
        
    frequencies_hz = data_frequencies.copy()
    frequencies_rad = frequencies_hz*2*np.pi

    _,H = scipy_frequency_response(poles,zeros,frequencies_rad)
    phase = np.angle(H)

    #   Transfer function (rad)
    fig,ax  = plt.subplots(2,layout='constrained')
    ax[0].plot(frequencies_rad,20*np.log10(H.real.__abs__()))

    ax[0].semilogx()
    ax[0].set_xlabel(r'$\omega$ (rad/s)')
    ax[0].set_ylabel(r'Response (dB)')
    ax[1].plot(frequencies_rad,phase)
    ax[1].semilogx()
    ax[1].set_xlabel(r'$\omega$ (rad/s)')
    ax[1].set_ylabel(r'Phase (rad)')
    plt.savefig(NICE_FIGURES.joinpath('frequency_response.png'),dpi=256)
    plt.close()

    #   Transfer function (Hz)
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
    plt.close()
    
    # #   Time domain impulse/step response
    # times = np.linspace(0,frequencies_hz.shape[0],frequencies_hz.shape[0])
    
    # t,impulse = scipy.signal.impulse((b_norm,a_norm),T=times)
    # print(impulse)
    # fig,(ax,ax1) = plt.subplots(2,layout='constrained')
    # ax.plot(t,impulse)
    # ax.set_xlabel('samples')
    # ax.set_ylabel('Impulse Response')

    # t,step = scipy.signal.step((b_norm,a_norm),T=times)

    # ax1.plot(t,step)
    # ax1.set_xlabel('samples')
    # ax1.set_ylabel('Step Response')
    # plt.savefig(NICE_FIGURES.joinpath('impulse_step_response.png'),dpi=256)
    # plt.close()


    #   Coefficient spectra

    fig,ax = plt.subplots(2,layout='constrained')
    ax[0].plot(np.abs(b_norm))
    ax[0].semilogy()
    ax[1].plot(np.abs(a_norm))
    ax[1].semilogy()
    ax[0].set_ylabel('Numerator Coefficents')
    ax[1].set_ylabel('Denomimator Coefficents')     
    plt.savefig(NICE_FIGURES.joinpath('coefficient_spectra.png')  )
    plt.close()

    #   Group delay

    fig,ax = plt.subplots(layout='constrained')
    _,group_delay = scipy.signal.group_delay((b_norm,a_norm),w=frequencies_rad,fs=2*np.pi*500)
    #group_delay = compute_group_delay(phase,frequencies_rad)
    inds = range(0,group_delay.shape[0])#group_delay <=0
    print(group_delay)
    ax.plot(frequencies_hz[:][inds],group_delay[inds])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (s/rad)')
    plt.savefig(NICE_FIGURES.joinpath('group_delay.png'),dpi=256)
    plt.close()


    #   Total Harmonic Distortion

    input_freqs = pulses[:,1]
    thd,w = compute_thd(X_spectra,Y_spectra,input_freqs,data_frequencies)

    mask = w==31.25


    fig,ax = plt.subplots()
    ax.plot(w,thd,'r*-')
    ax.set_title(f'THD at 31.25Hz = {np.mean(thd[mask]):02f} +- {np.std(thd[mask]):04f} dB')
    ax.set_ylabel('THD (dB)')
    ax.set_xlabel('Input Freq (Hz)')

    plt.savefig(NICE_FIGURES.joinpath('THD_observed_diff.png'))
    plt.close()


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
    plt.close()

    #   FOr SFDR we need to look for the highest other peak, relative to the fundamental. Should be +ve
  
    return
def out_plots(pandz,coeffs_per_stage,frequencies,X_spectra,Y_spectra,pulses,FIGPATH):

    poles_final,zeros_final = pandz[:,0],pandz[:,1]
    if coeffs_per_stage is not None:
        data_reconst = apply_stages(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,coeffs_per_stage,True)
    #data_reconst = g_scipy(poles_final,zeros_final,X_spectra,Y_spectra,frequencies,True)

    fig,ax = plt.subplots(2,layout='constrained')
    H = calculate_transfer_function(poles_final,zeros_final,2*np.pi*frequencies)
    ax[0].plot(frequencies,H.real.__abs__()/(2*np.pi) + 1e-5)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel(r'$\mathfrak{R}$')
    ax[1].set_ylabel(r'$\mathfrak{I}$')
    ax[1].plot(frequencies,H.imag.__abs__())
    #ax[0].loglog()
    plt.savefig(f'{FIGPATH}/Transfer_function.png')
    plt.close()

    fig,ax = plt.subplots(layout='constrained')
    ax.plot(poles_final.real,poles_final.imag,'x',label='Poles')
    ax.plot(zeros_final.real,zeros_final.imag,'o',label='Zeros')
    x = np.linspace(-1,1,100)
    y = np.sqrt(1-x**2)
    ax.plot(x,y,'k--')
    ax.plot(x,-y,'k--')
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

    #.42.30
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
    ax1.set_xlabel(r'$\omega$ (rad/s)')
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