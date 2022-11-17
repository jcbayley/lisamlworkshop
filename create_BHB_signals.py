import bilby
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lisagwresponse import GalacticBinary, ReadStrain
import pytdi
import pytdi.michelson
import os
import pickle


def create_polarisations(injection_parameters, sampling_frequency, duration, reference_time, approximant="IMRPhenomD", minimum_frequency=1e-4):
    """ Generates the polarisations of signal to be combined later.
    """

    if sampling_frequency>4096:
        print('EXITING: bilby doesn\'t seem to generate noise above 2048Hz so lower the sampling frequency')
        exit(0)

    # compute the number of time domain samples
    Nt = int(sampling_frequency*duration)
    
    # define the start time of the timeseries
    start_time = reference_time-duration/2.0

    times = np.arange(start_time, start_time + duration, 1./sampling_frequency)
    
    # Fixed arguments passed into the source model 
    waveform_arguments = dict(waveform_approximant=approximant,
                              reference_frequency=1e-3, 
                              minimum_frequency=minimum_frequency,
                              maximum_frequency=sampling_frequency/2.0)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        start_time=start_time)
    
    # extract waveform from bilby
    waveform_generator.parameters = injection_parameters
    waveform_polarisations = waveform_generator.frequency_domain_strain()
    time_signal = waveform_generator.time_domain_strain()
    return times, time_signal

def get_psd(psd_file):
    # load noisy psd file generated from lisanode
    psdNoiseLN15 = psd.Series.from_file(fNNoiseLN15+":X",skiprows=200)
    psdNoiseLN15[0].compute(nperseg=200000)
    return psdNoiseLN15


def get_resp_and_tdi(save_dir, index, priors, start_time, duration, data_length, sampling_frequency, reference_time, minimum_frequency, skipped=200):
    """ Get the 6 responses to the GW and then compute the X Y and Z tdis"""
    # sample from the prior by set m1=m2 for simplicity
    injection_pars = priors.sample()
    injection_pars["mass_2"] = injection_pars["mass_1"]

    # generate the polarisations using IMRPhenomD
    times, tim_pol = create_polarisations(injection_pars, sampling_frequency, duration, reference_time, minimum_frequency=minimum_frequency)

    # set the sky positions as ra and dec 
    # (technically the ra and dec are not lambda and beta but for these purposes it doesnt matter as it will be randomly placed on the sky either way)
    gw_lambda = injection_pars["ra"]
    gw_beta = injection_pars["dec"]
    # gw_beta and gw_lambda ar ethe sky position in ecliptic coords
    # define the start time of the responses (place the start time so coalescence is 0.1*datalength from the end) 
    data_start = start_time+(duration - data_length/sampling_frequency) + 0.1*data_length/sampling_frequency
    #setup the response function
    data = ReadStrain(times, tim_pol["plus"], tim_pol["cross"], gw_beta=gw_beta, gw_lambda=gw_lambda,orbits="./data/esa-orbits-1-0-2.h5", size = data_length, dt = 1/sampling_frequency, t0 = data_start) 

    # compute the response for each of the LINKS
    resp = data.compute_gw_response(np.arange(data_start, data_start+data_length*1./sampling_frequency, 1./sampling_frequency),data.LINKS)

    resp_dir = os.path.join(save_dir, "response")
    if not os.path.isdir(resp_dir):
        os.makedirs(resp_dir)

    # write the response from each example to a file to be loaded by pytdi
    data.write(path=os.path.join(resp_dir, f"test_gw_{index}.h5"),mode="w")

    # read the response into pytdi 
    # skip some samples to avoid edge effects
    tdi = pytdi.Data.from_gws(os.path.join(resp_dir, f"test_gw_{index}.h5"), "./data/esa-orbits-1-0-2.h5", skipped=skipped)

    # compute the X Y and Z tdi
    X2 = pytdi.michelson.X2.build(**tdi.args)
    X2_data = X2(tdi.measurements)

    Y2 = pytdi.michelson.Y2.build(**tdi.args)
    Y2_data = Y2(tdi.measurements)

    Z2 = pytdi.michelson.Z2.build(**tdi.args)
    Z2_data = Z2(tdi.measurements)

    return injection_pars, (X2_data, Y2_data, Z2_data), resp, tim_pol

def PSD_Noise_components(fr, sqSnoise):
    """ taken from https://gitlab.in2p3.fr/LISA/lisa_sensitivity_snr/-/blob/master/lisa_sensitivity_snr.ipynb"""
    [sqSacc_level, sqSoms_level] = sqSnoise
    # sqSacc_level: Amplitude level of acceleration noise [3e-15]
    # sqSoms_level: Amplitude level of OMS noise [15e-12]
    
    c = 3e8
    ## Armlength in seconds
    ### Acceleration noise
    Sa_a = sqSacc_level**2 *(1.0 +(0.4e-3/fr)**2)*(1.0+(fr/8e-3)**4)
    Sa_d = Sa_a*(2.*np.pi*fr)**(-4.)
    Sa_nu = Sa_d*(2.0*np.pi*fr/c)**2

    ### Optical Metrology System
    Soms_d = sqSoms_level**2 * (1. + (2.e-3/fr)**4)
    Soms_nu = Soms_d*(2.0*np.pi*fr/c)**2
    
    return [Sa_nu, Soms_nu]

def PSD_Noise_X15(fr, sqSnoise):
    """ taken from https://gitlab.in2p3.fr/LISA/lisa_sensitivity_snr/-/blob/master/lisa_sensitivity_snr.ipynb """
    L_m = 2.5e9
    c = 3e8
    L = L_m/c
    [Sa_nu,Soms_nu] = PSD_Noise_components(fr, sqSnoise)
    phiL = 2*np.pi*fr*L
    return 16*( np.sin(phiL))**2 * (Soms_nu + Sa_nu*(3+np.cos(phiL)) )
    
def PSD_Noise_X20(fr, sqSnoise):
    """ taken from https://gitlab.in2p3.fr/LISA/lisa_sensitivity_snr/-/blob/master/lisa_sensitivity_snr.ipynb """
    [Sa_nu,Soms_nu] = PSD_Noise_components(fr, sqSnoise)
    L_m = 2.5e9
    c = 3e8
    L = L_m/c
    phiL = 2*np.pi*fr*L
    return 64 * (np.sin(phiL))**2 * ( np.sin(2*phiL))**2 * (Soms_nu + Sa_nu*(3+np.cos(2*phiL)) )

def add_noise_to_tdi(tdis, sampling_frequency):

    # get the frequencies from the input data and sampling frequency
    noise_freq = np.fft.rfftfreq(np.shape(tdis)[2], 1./sampling_frequency)

    # compute a theoretical psd for LISA
    #sqSnoise_SciRD = [3e-15,15e-12]
    sqSnoise_SciRD = [3e-14,15e-12]
    an_noise = PSD_Noise_X15(noise_freq,sqSnoise_SciRD)

    fshape = np.array(np.shape(tdis))
    Nt = fshape[2]
    dt = 1./sampling_frequency
    # half the length to generate the one sided frequency series
    fshape[2] = fshape[2]//2 + 1
    # calculate standard deviation for the frequency domain given a psd
    sigma = np.sqrt(Nt*an_noise/(2*dt))
    # add noise in the complex frequency domain
    noise_fs_noise = 0.5*sigma*(np.random.normal(0,1,size=fshape) +1j*np.random.normal(0,1,size=fshape))
    noise_fs_signal = 0.5*sigma*(np.random.normal(0,1,size=fshape) +1j*np.random.normal(0,1,size=fshape))

    coloured_ts_noise = np.fft.irfft(np.nan_to_num(noise_fs_noise))
    coloured_ts_signal= np.fft.irfft(np.nan_to_num(noise_fs_signal))

    # noise is additive to add the tdi representations to the noise
    output_data_signal = np.array(tdis) + coloured_ts_signal

    return output_data_signal, coloured_ts_noise, an_noise

def whiten_data(tdis, psd):

    # get frequency spectrum of data and divide by the sqrt(psd) to whiten the data
    Nt = len(tdis)
    tdift = np.nan_to_num(np.fft.rfft(tdis, axis=-1))
    # normalise so time series has a sdtandard deviation of 1
    whiten_tdi = np.sqrt(Nt)*np.fft.irfft(np.nan_to_num(tdift/np.sqrt(psd)))

    return whiten_tdi

def save_all_signals(save_dir, num_signals, save_parameters, data_pars, start_index=0, add_noise=True):

    all_injection_parameters = []
    save_injection_parameters = []

    # define the prior based on a prior file
    priors = bilby.gw.prior.BBHPriorDict(filename="./bbh.prior")

    all_tdi = []
    all_resp = []

    # compute the responses and tdi representations for many examples
    for i in range(num_signals):
        injection_parameters, tdi, resp, tim_pol = get_resp_and_tdi(save_dir, start_index + i, priors, data_pars["start_time"], data_pars["duration"], data_pars["data_length"], data_pars["sampling_frequency"], data_pars["reference_time"], data_pars["minimum_frequency"],skipped=data_pars["skipped"])
        all_injection_parameters.append(injection_parameters)
        save_injection_parameters.append([1] + [injection_parameters[key] for key in save_parameters])
        all_tdi.append(tdi)
        all_resp.append(resp)

    # save the responses and tdis to a pickle file
    with open(os.path.join(save_dir, f"tdis_{start_index}_{num_signals}.pkl"),"wb") as f:
        pickle.dump([save_injection_parameters,all_tdi], f)

    with open(os.path.join(save_dir, f"resp_{start_index}_{num_signals}.pkl"),"wb") as f:
        pickle.dump([save_injection_parameters,all_resp], f)

    if add_noise:
        # add noise the tsi and generate noise only tdi
        signal_data, noise_data, an_noise = add_noise_to_tdi(all_tdi, data_pars["sampling_frequency"])

        # make some labels and injections parameters for the noise only case
        noise_injection_parameters = np.zeros(np.shape(save_injection_parameters))
        # set all noise only parameters to nan
        noise_injection_parameters[:,1:] = np.nan

        # join the noise only and noise + signal datasets
        pars = np.append(save_injection_parameters, noise_injection_parameters, axis = 0)
        data = np.append(signal_data, noise_data, axis = 0)

        whitened_data = whiten_data(data, an_noise)

        # save to file
        with h5py.File(os.path.join(save_dir, f"sig_noise_data_{start_index}_{2*num_signals}.h5"),"w") as data_file:
            data_file.create_dataset('whitened_data', data=whitened_data)
            data_file.create_dataset('parameters', data=pars)


if __name__ == "__main__":

    # load lisa orbits file to get the start times 
    orbits = "./data/esa-orbits-1-0-2.h5"
    save_dir = "./data3/"
    # define parameters to save 
    save_parameters = ["mass_1", "mass_2", "luminosity_distance", "dec", "ra"]

    with h5py.File('./data/esa-orbits-1-0-2.h5') as f:
        orbits_t0 = f.attrs['t0']
        orbits_duration = f.attrs['tduration']
        orbits_times = [t.strip("[").strip("]").strip("\n") for t in f.attrs["t"].split(" ")]

    # set duration of signal to large value 
    data_pars = {"duration":10000000}
    # sampling frequency set to 1e-3 to save on the amount of data generated
    data_pars["sampling_frequency"] = 0.001
    data_pars["reference_time"] = orbits_t0 + data_pars["duration"] 
    data_pars["start_time"] = data_pars["reference_time"] - data_pars["duration"]/2.0
    # minimum frequency set to be at lower edge of band
    data_pars["minimum_frequency"] = 8e-5
    data_pars["skipped"] = 200
    # define the output data length to be 1024 (add skipped to keep output at 1024 samples)
    data_pars["data_length"] = 1024 + data_pars["skipped"]
   
    save_all_signals(save_dir, 10000, save_parameters, data_pars = data_pars, add_noise=True)




