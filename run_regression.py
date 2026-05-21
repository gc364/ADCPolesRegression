#!/usr/bin/python
from pathlib import Path
from bfgs import main_lbfgs
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description='Run the regression algorithm for the ADC transfer function')
    parser.add_argument('working_directory',type=Path,help='Working directory for outputs')
    parser.add_argument('data_directories',type=str,help='Comma seperated list of the data directories (T-Rex working directories)')
    parser.add_argument('channel',type=str,help='The channel to analyse (e.g. ch00)')
    parser.add_argument('stages',type=str,help='Comma seperated list of the type of filter (LINEAR or MIN) for each stage')
    parser.add_argument('num_coeffs',type=str,help='Comma seperated list of the number of coefficients for each stage')
    parser.add_argument('--datalogger_sample_rate',type=int,help='The sample rate the data was collected at in Hz',required=False,default=250)
    parser.add_argument('--sinc_decimation',type=str,help='Comma seperated list of sinc decimations',required=False,default=None)
    parser.add_argument('--plots_only',action='store_true',help='Only generate plots?',required=False)
    parser.add_argument('--ftol',type=float,help='Function change threshold for termination',required=False,default=1e-5)
    parser.add_argument('--nepochs',type=int,help='Maximum number of function evaluations',required=False,default=200)
    parser.add_argument('--nfrequency',type=int,help='Number of frequency points in the data fft. Higher means a higher frequency resolution',required=False,default=12000)
    
    args = parser.parse_args()
     
    workdir = args.working_directory
    workdir = workdir.joinpath(f'{args.channel}')
    workdir.mkdir(exist_ok=True)
    workdir.joinpath('figures/output').mkdir(exist_ok=True,parents=True)
    workdir.joinpath('results').mkdir(exist_ok=True)

    #   Datasets to load
    paths = [Path(direc) for direc in args.data_directories.split(',')]
    #   coeffs per stage is the number of coeffs we seek, so for minphase it'll be 2 times this and 4 times this for linear
    coeffs_per_stage = [int(c) for c in args.num_coeffs.split(',')]
    #   The type of filter per stage
    phases_per_stage = args.stages.split(',')#['MIN']
    #   Decimations for the sinc filters, None if no sinc filters required
    sinc_dec = [int(sd) for sd in args.sinc_decimation] if args.sinc_decimation is not None else None
    #   If True, will create a new optimisation, else just reruns plotting
    new_optimise = not args.plots_only

    main_lbfgs(paths,coeffs_per_stage,phases_per_stage,workdir,args.nepochs,new_optimise,args.ftol,sinc_dec,args.nfrequency,args.channel,args.datalogger_sample_rate)

    exit()