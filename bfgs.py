import numpy as np
from pathlib import Path
from filter_coeffs import *
from scipy.optimize import minimize,fmin_l_bfgs_b,OptimizeResult
from shared import *
from types import SimpleNamespace
import json
import datetime
from xml.etree import ElementTree as gfg
import obspy

def objective_lbfgs(m,nzeros,X_spectra,Y_spectra,frequencies,phases_per_stage,
                    scipy=False,coeffs_per_stage=None,set_poles=None):

    zeros_list = []
    poles_list = []
  
    coeffs_per_stage_c = deepcopy(coeffs_per_stage)
    for i,(phase,nc) in enumerate(zip(phases_per_stage,coeffs_per_stage)):
        zeros = m[i*nc:(i+1)*nc]*np.exp(1j* m[(nzeros//2)+(i*nc):(nzeros//2)+((i+1)*nc)]) 

        if phase == 'LINEAR':
            zeros = np.concatenate([zeros,1/zeros])
            coeffs_per_stage_c[i] *=2

        if set_poles is None:
            poles = m[nzeros:(3*nzeros)//2]*np.exp(1j*m[(3*nzeros)//2:])
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
        l = -apply_stages(poles,zeros,X_spectra,Y_spectra,frequencies,coeffs_per_stage_c)
    elif scipy == False:
        l  = -g(poles,zeros,X_spectra,Y_spectra,frequencies)
    else:
        l = -g_scipy(poles,zeros,X_spectra,Y_spectra,frequencies)
    
    return np.sqrt(l.__abs__()@l.__abs__().T)



def callback_lbfgs(intermediate_result):
    #print(f'Value: {intermediate_result.x}')
    print(f'Iteration Loss: {intermediate_result.fun}')
    
    return


def load_datasets(paths:list[Path],workdir,nfrequency,channel,sinc_dec=None):
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
    for datadir in paths:
        nc_path = list(datadir.joinpath('processed/netcdf').glob('*.nc'))[0]
        pulses_path = datadir.joinpath(f'pulses_{channel}.txt')
        if not pulses_path.is_file():
            raise FileNotFoundError(f'pulses_{channel}.txt is not in the top level of the data directory')
        X_spectra_i,Y_spectra_i,frequencies  = load_xy(pulses_path,nc_path,workdir,nfrequency,channel)
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
    i=0
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
            if (bounds.phase == 'MIN') or (bounds.phase is None):
                phi = np.linspace(0,np.pi,nz)
                r = np.ones(nz)-(0.1*(i+1))
                zeros_list_r.append(r)
                zeros_list_phi.append(phi)
            elif bounds.phase=='LINEAR':
                phi = np.linspace(0,np.pi,nz)
                r = np.ones(nz)-(0.1*i)
                zeros_list_r.append(r)
                zeros_list_phi.append(phi)
        
        if set_poles is None:
            poles_list.append(np.concatenate([
                                    np.random.uniform(bounds.poles_r_min,bounds.poles_r_max,nz),
                                    np.random.uniform(bounds.poles_phi_min,bounds.poles_phi_max,nz)
                                    ]))
        i+=1
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

    set_poles_size = 0
    bounds_list = []
    
    for phase in phases_per_stage:
        if type=='FIR':
            if phase == 'MIN':
                set_poles_size += nz
            elif phase == 'LINEAR':
                set_poles_size += nz
        elif 'IIR':
            raise NotImplementedError("We can't do IIR Filters")
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
   
    for i,(phase,nc) in enumerate(zip(phases_per_stage,coeffs_per_stage)):
        if phase=='MIN' or phase == None:
            if set_poles is not None:
                poles_final = set_poles[i*nc:(i+1)*nc]*np.exp(1j*set_poles[nz//2 + i*nc:nz//2 + (i+1)*nc]) 
            else:
                poles_final = m_post[nz:(3*nz)//2]*np.exp(1j*m_post[(3*nz)//2:])

            zeros_final = m_post[i*nc:(i+1)*nc]*np.exp(1j*m_post[nz//2 + i*nc:nz//2 + (i+1)*nc]) 


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

def write_metadata(workdir,phases_per_stage,coeffs_per_stage,res:OptimizeResult,ftol,nfrequency,sinc_dec,data_dirs,channel,total_time,datalogger_sample_rate):
    number_of_stages = len(phases_per_stage)
    file = workdir.joinpath('metadata.json')
    file.touch(exist_ok=True)
    nc_files  = [list(dd.joinpath('processed/netcdf').glob('*.nc'))[0] for dd in data_dirs]

    metadata = {}
    metadata.update({'data_files':[str(dd) for dd in nc_files]})
    metadata.update({'datalogger_sample_rate':datalogger_sample_rate})
    metadata.update({'channel':channel})
    metadata.update({'number_of_stages':number_of_stages})
    metadata.update({'nfrequency':nfrequency})
    metadata.update({'sinc_filters':sinc_dec if sinc_dec is not None else 'None'})

    metadata.update({'regression_success':res.success})
    metadata.update({'regression_message':res.message})
    metadata.update({'regression_time_seconds':total_time})

    metadata.update({'regression_ftol':ftol})
    metadata.update({'regression_final_loss':res.fun})
    metadata.update({'regression_number_iterations':res.nit})
    for stage in range(number_of_stages):
        stage_meta = {}
        stage_meta.update({
            'filter_type':phases_per_stage[stage],
            'number_of_zeros':2*coeffs_per_stage[stage] if phases_per_stage[stage]=='MIN' else 4*coeffs_per_stage[stage]
        })
        metadata.update({f'stage_{stage}':stage_meta})

    with open(file,'w+') as fd:
        json.dump(metadata,fd,indent=1)
    
    return
def create_top_station_xml(channel,datalogger_s_r):
    xml_file = gfg.Element('FDSNStationXML',attrib={'xmlns':"http://www.fdsn.org/xml/station/1",
                                                    'schemaVersion':"1.2"})
    src = gfg.SubElement(xml_file,'Source')
    src.text  = 'test'
    src = gfg.SubElement(xml_file,'Created')
    src.text = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    Net = gfg.SubElement(xml_file,'Network',attrib={'code':'XX'})
    Sta = gfg.SubElement(Net,'Station',attrib={'code':'ABCD'})

    lat =  gfg.SubElement(Sta,'Latitude',attrib={'unit':'DEGREES'})
    lat.text = '0.0'
    long =  gfg.SubElement(Sta,'Longitude',attrib={'unit':'DEGREES'})
    long.text = '0.0'
    el =  gfg.SubElement(Sta,'Elevation')
    el.text = '0.0'
    site = gfg.SubElement(Sta,'Site')
    nm = gfg.SubElement(site,'Name')
    nm.text = 'None'

    Chan = gfg.SubElement(Sta,'Channel',attrib={'code':channel,'locationCode':'00'})
    lat =  gfg.SubElement(Chan,'Latitude',attrib={'unit':'DEGREES'})
    lat.text = '0.0'
    long =  gfg.SubElement(Chan,'Longitude',attrib={'unit':'DEGREES'})
    long.text = '0.0'
    el =  gfg.SubElement(Chan,'Elevation')
    el.text = '0.0'
    dep =  gfg.SubElement(Chan,'Depth')
    dep.text = '0.0'

    az =  gfg.SubElement(Chan,'Azimuth')
    az.text = '0.0'

    dip =  gfg.SubElement(Chan,'Dip')
    dip.text = '0.0'
    

    sr = gfg.SubElement(Chan,'SampleRate')
    sr.text = str(datalogger_s_r)

    return Chan,xml_file
def create_station_xml_response(pandz,workdir,channel,datalogger_s_r):
    
    zeros = pandz[:,1].copy()
    poles  = pandz[:,0].copy()

    chan,xml_file = create_top_station_xml(channel,datalogger_s_r)
  
    resp = gfg.SubElement(chan,'Response')
    sens = gfg.SubElement(resp,'InstrumentSensitivity')
    v = gfg.SubElement(sens,'Value')
    v.text = '1'
    v = gfg.SubElement(sens,'Frequency')
    v.text = str(datalogger_s_r//2)

    units = gfg.SubElement(sens,'InputUnits')

    un = gfg.SubElement(units,'Name')
    un.text = 'V'
    un = gfg.SubElement(units,'Description')
    un.text = 'Volts'
    units = gfg.SubElement(sens,'OutputUnits')
    un = gfg.SubElement(units,'Name')
    un.text = 'counts'
    un = gfg.SubElement(units,'Description')
    un.text = 'Digital Counts'

    stage = gfg.SubElement(resp,'Stage',attrib={'number':'1'})
    PZ = gfg.SubElement(stage,'FIR')

    units = gfg.SubElement(PZ,'InputUnits')

    un = gfg.SubElement(units,'Name')
    un.text = 'V'
    un = gfg.SubElement(units,'Description')
    un.text = 'Volts'
    units = gfg.SubElement(PZ,'OutputUnits')
    un = gfg.SubElement(units,'Name')
    un.text = 'counts'
    un = gfg.SubElement(units,'Description')
    un.text = 'Digital Counts'
    un = gfg.SubElement(PZ,'Symmetry')
    un.text = 'NONE'    

    num,den  = to_coefficents(poles,zeros)
    i=0
    for n in num.real:
        c = gfg.SubElement(PZ,'NumeratorCoefficient',attrib={'number':str(i)})
        c.text = str(n)
        
        i+=1
    
    dec = gfg.SubElement(stage,'Decimation')
    isr = gfg.SubElement(dec,'InputSampleRate')
    isr.text = str(datalogger_s_r)

    fac = gfg.SubElement(dec,'Factor')
    fac.text = '1'

    fac = gfg.SubElement(dec,'Offset')
    fac.text = '0'

    fac = gfg.SubElement(dec,'Delay')
    fac.text = '0'

    fac = gfg.SubElement(dec,'Correction')
    fac.text = '0'
 

    gain = gfg.SubElement(stage,'StageGain')
    n = gfg.SubElement(gain,'Value')
    n.text = '1'
    n = gfg.SubElement(gain,'Frequency')
    n.text = str(datalogger_s_r//2)
    tree = gfg.ElementTree(xml_file)
    with open (workdir.joinpath('results/StationXML.xml'), "wb") as fd :
        tree.write(fd)

    return

def station_xml_test(workdir):
    try:
        inv = obspy.read_inventory(workdir.joinpath('results/StationXML.xml'))
        inv.plot_response(0.001,'DEF',show=False,unwrap_phase=True)
    except:
        print('!*30')
        raise UserWarning('Filter Failed to Load from StationXML !!!')
        print('!*30')
    return
def main_lbfgs(paths,coeffs_per_stage,phases_per_stage,workdir,nepochs,new_optimise,ftol,sinc_dec,nfrequency,channel,datalogger_sample_rate):
    num_workers = 1
    figpath = workdir.joinpath('figures/output')
    f_type='FIR'    #   Hardcoded as it can't handle IIR at all
    X_spectra,Y_spectra,frequencies,pulses = load_datasets(paths,workdir,nfrequency,channel,sinc_dec)
    print('Datasets Loaded')

    if new_optimise:
        m0,bounds,set_poles,nz = get_system(f_type,phases_per_stage,coeffs_per_stage)
        print('Got Settings')
        print('Optimsing...')
        start_time = datetime.datetime.now()
        print(f'Start Time: {start_time}')
        nz =  sum(coeffs_per_stage)*2

        res = minimize(
                objective_lbfgs,
                args = (nz,X_spectra,Y_spectra,frequencies,phases_per_stage,True,coeffs_per_stage,set_poles),
                x0=m0,
                method='L-BFGS-B', 
                options={
                         'maxiter':nepochs,
                         'maxfun':1e6,  #   This is really big as it takes at least nz evaluations to compute 1 Jacobian
                         'ftol':ftol,
                         'workers':num_workers
                         },
                bounds=bounds,
                callback=callback_lbfgs,
                
                )
    
        print('Done!')
        end_time = datetime.datetime.now()
        print(f'End Time: {end_time}')
        total_time = end_time.timestamp()-start_time.timestamp()
        time_hrs = np.floor(total_time/3600)
        if time_hrs >0:
            time_mins = (total_time%(time_hrs*3600))/60
        else:
            time_mins = (total_time/3600)/60
        if np.floor(time_mins)>0:
            time_secs  = (time_mins%np.floor(time_mins))*60
        else:
            time_secs = total_time
        print(f'Total Time: {int(time_hrs)}:{int(np.floor(time_mins))}:{round(time_secs,3)}')
        print(res)
        m_post = res.x

        pandz = sort_mpost(m_post,nz,set_poles,phases_per_stage,coeffs_per_stage)
        np.save(workdir.joinpath('results/PolesandZeros.npy'),pandz)
        np.save(workdir.joinpath('results/PolesandZeros_Dimensionless.npy'),pandz/(datalogger_sample_rate*2*np.pi))
        print('Plotting')
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
        write_metadata(workdir,phases_per_stage,coeffs_per_stage,res,ftol,nfrequency,sinc_dec,paths,channel,total_time,datalogger_sample_rate)

    else:
        pandz =  np.load(workdir.joinpath('results/PolesandZeros.npy'))
        poles_final = pandz[:,0]
        zeros_final = pandz[:,1]
    create_station_xml_response(pandz,workdir,channel,datalogger_sample_rate)
    station_xml_test(workdir)
    out_plots(pandz,coeffs_per_stage,phases_per_stage,frequencies,X_spectra,Y_spectra,pulses,figpath,datalogger_sample_rate)
    create_nice_figures(poles_final,zeros_final,workdir,frequencies,coeffs_per_stage,phases_per_stage,datalogger_sample_rate)
    create_fir_filter_PZs(workdir)
    print('Regression complete !')
    return

