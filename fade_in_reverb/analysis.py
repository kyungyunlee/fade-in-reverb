import os 
import numpy as np 
from sklearn.cluster import KMeans
import scipy
import torch
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from DecayFitNet.python.toolbox.core import  decay_kernel, schroeder_to_envelope, PreprocessRIR, decay_model, discard_last_n_percent, FilterByOctaves
import yaml
config = yaml.safe_load(open(os.path.abspath("/Users/kyungyunlee/dev/fade-in-reverb/fade_in_reverb/config.yaml")))


print (config)

def load_common_decay_times(filepath, rirs, n_slopes) : 
    """
    filepath : 
    rirs : shape=(n_rirs, num_samples)
    n_slopes : 
    """
    if not os.path.exists(filepath) : 
        print (f"{filepath} does not exist. Need to compute common decay times")
        if not os.path.exists(os.path.dirname(filepath)): 
            os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        common_decay_times = compute_common_decay_times(rirs, n_slopes)
        # save 
        np.save(filepath, common_decay_times)

    else : 
        # load 
        print ("Loading pre-computed common decay times")
        common_decay_times = np.load(filepath)
    return common_decay_times 


def compute_common_decay_times(rirs, n_slopes) : 
    """
    rirs : shape=(n_rirs, num_samples)
    n_slopes : 
    """
    common_decay_times =  [] 

    for bIdx in range(len(config['f_bands'])) :
        curr_band = config['f_bands'][bIdx]

        # # Predict exponential model parameters
        # DecayFitNet can estimate maximum of 3 slopes at the moment.
        decayfitnet = DecayFitNetToolbox(n_slopes=3, sample_rate=config['sample_rate'], filter_frequencies=[curr_band])
        estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(rirs, analyse_full_rir=True)
        tVals_standard, aVals_standard, nVals_standard = estimated_parameters_decayfitnet

        # Determine common decay times from k-means method
        nonZeroT = tVals_standard[tVals_standard > 0] 
        nonZeroT =  nonZeroT.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_slopes, random_state=0, max_iter=10000, n_init=5, algorithm='elkan').fit(nonZeroT)
        # curr_band_common_decay_times = kmeans.cluster_centers_[:, 0]
        tIdx2ClusterIdx = kmeans.labels_

        histResolution = 0.05 
        histMin = np.floor(min(nonZeroT)/histResolution)*histResolution
        histMax = np.ceil(max(nonZeroT)/histResolution)*histResolution
        histEdges = np.arange(histMin, histMax, histResolution) 
        clusteredTVals = defaultdict(list)
        nonZeroT = nonZeroT[:, 0]

        curr_band_common_decay_times = np.zeros((n_slopes,))
        for mIdx in range(n_slopes) : 

            # print (nonZeroT[tIdx2ClusterIdx == mIdx] ) 
            clusteredTVals[mIdx] = nonZeroT[tIdx2ClusterIdx == mIdx] 
            hist, bin_edges = np.histogram(clusteredTVals[mIdx], histEdges)
            # print (hist) 
            maxCountIdx = np.argmax(hist)
            # print (maxCountIdx) 
            # print (histEdges[maxCountIdx: maxCountIdx+1])
            # commonDecayTimes[mIdx] =histEdges[maxCountIdx]
            curr_band_common_decay_times[mIdx] = np.mean(histEdges[maxCountIdx: maxCountIdx+2]) 

        common_decay_times.append(curr_band_common_decay_times)
        
        
        print (curr_band_common_decay_times)
    
    common_decay_times = np.array(common_decay_times)
    common_decay_times = np.sort(common_decay_times, 1)

    return common_decay_times 


def get_envelope(rir, window_size):
    # RMS 
    rms_list = [] 
    for i in range(rir.shape[0]//window_size): 
        curr_rms = np.sqrt(np.mean(rir[i*window_size:i * window_size + window_size] **2))
        rms_list.append(curr_rms)
    rms_list = np.array(rms_list)

    a1 = 0.3
    b = [1-a1]
    a = [1, -a1] 
    rms_list = scipy.signal.lfilter(b, a, rms_list)

    
    return rms_list 


def load_envelope_fit_result(filepath, rirs, common_decay_times, multichannel, plot=False) : 
    if not os.path.exists(filepath) : 
        print (f"{filepath} does not exist. Need to perform fitting.")
        if not os.path.exists(os.path.dirname(filepath)): 
            os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        pos_fit_result, neg_fit_result, all_original_env = envelope_fit(rirs, common_decay_times, multichannel, plot)

        result_dict = {"pos_fit_result" : pos_fit_result,
                       "neg_fit_result": neg_fit_result,
                       "all_original_env": all_original_env}
        # res = np.array([pos_fit_result, neg_fit_result, all_original_env])
        # np.save(filepath, res)
        pickle.dump(result_dict, open(filepath, 'wb'))

        print ("Saved to", filepath)
        
    else : 
        print ("Loading pre-computed result from", filepath)
        # result = np.load(filepath)
        result = pickle.load(open(filepath, 'rb'))
        pos_fit_result = result['pos_fit_result']
        neg_fit_result = result['neg_fit_result']
        all_original_env = result['all_original_env']

    return pos_fit_result, neg_fit_result, all_original_env



def envelope_fit(rirs, common_decay_times, multichannel, plot=False) : 

    """ Given RIRs, perform least squares fit on the envelope with the common slope times. 
    rirs : shape=(n_rirs, num_samples) if not multichannel, else (n_rirs, n_channels, num_samples)
    """

    # rirs = rirs / np.max(np.abs(rirs)) * 0.999 

    if not multichannel : 
        # Add one dimension 
        rirs = rirs[:, np.newaxis, :]

    n_rirs, n_channels, L = rirs.shape

    # Downsampling length for 
    if L == 48000 or L == 96000: 
        downSampleLength = 240 
    elif L == 40000 : 
        downSampleLength = 200 
    else : 
        print ("RIR size is not valid:", L)
        exit()
    
    
    downSampleRate = L // downSampleLength
    window_size = downSampleRate 

    ds_start_index = 2

    n_slopes = len(common_decay_times[0])


    # shape = (num RIRs , num octaves, 3 slopes+ noise) 
    all_original_env = np.zeros((n_rirs, n_channels, len(config['f_bands']), downSampleLength - ds_start_index))
    neg_fit_result = np.zeros((n_rirs, n_channels, len(config['f_bands']), n_slopes+1))
    pos_fit_result = np.zeros((n_rirs, n_channels, len(config['f_bands']), n_slopes+1))
    # else : 
    #     all_original_env = np.zeros((n_rirs, len(config['f_bands']), downSampleLength - ds_start_index))
    #     neg_fit_result = np.zeros((n_rirs, len(config['f_bands']), n_slopes+1))
    #     pos_fit_result = np.zeros((n_rirs, len(config['f_bands']), n_slopes+1))
    

    for bIdx in range(len(config['f_bands'])) : 
        curr_common_decay_times = common_decay_times[bIdx]
        print (curr_common_decay_times)
            
        # Filter signal by octave
        filterbank = FilterByOctaves(order=6, sample_rate=config['sample_rate'], backend='scipy',
                                                center_frequencies=[config['f_bands'][bIdx]])
        
        for rIdx in range(n_rirs): 

            for cIdx in range(n_channels) : 
                rir = rirs[rIdx][cIdx]

                # Perform octave filtering at the current octave 
                octave_filtered_rir = filterbank(torch.FloatTensor(rir))[0]
                octave_filtered_rir = octave_filtered_rir.numpy()

                octave_filtered_rir_norm_factor = np.max(np.abs(octave_filtered_rir)) 
                octave_filtered_rir = octave_filtered_rir / octave_filtered_rir_norm_factor

                # Make time axis (downsampled version and full version)
                timeAxis_ds = np.linspace(0, L / config['sample_rate'], downSampleLength- ds_start_index) 
                timeAxis_fullLength = np.linspace(0, L/config['sample_rate'], L)
                
                # Get the exponentials 
                envelopeTimes = 2 * curr_common_decay_times
                envelopes = decay_kernel(envelopeTimes, timeAxis_ds)
                envelopes_fullLength = decay_kernel(envelopeTimes, timeAxis_fullLength)
                
                # The noise part is just ones 
                envelopes[:, -1] = np.ones_like(envelopes[:, -1]) 
                envelopes_fullLength[:, -1] = np.ones_like(envelopes_fullLength[:, -1]) 

                # RMS for later gain matching 
                original_rms = np.sqrt(np.mean(octave_filtered_rir**2))

                # Original RIR's envelope 
                original_env_ds = get_envelope(octave_filtered_rir, window_size)
                original_env_ds = original_env_ds[ds_start_index:]

                # Save for later 
                all_original_env[rIdx][cIdx][bIdx] = original_env_ds * octave_filtered_rir_norm_factor

                # Compute EDF for plotting later 
                rir_preprocessing = PreprocessRIR(sample_rate=config['sample_rate'], filter_frequencies=[config['f_bands'][bIdx]])
                edf, __ = rir_preprocessing.schroeder(rir, analyse_full_rir=True) # normalized EDFs 
                edf = edf.squeeze(0).squeeze(0)
                edf = edf.numpy()
                edf_ds = scipy.signal.resample_poly(edf, up=1, down=downSampleLength)

                # decayKernel_fullLength = decay_kernel(curr_common_decay_times, timeAxis_fullLength)
                # decayKernel_ds = decay_kernel(curr_common_decay_times, timeAxis_ds)

                # Initial values for fitting 
                x0_pos = np.zeros((n_slopes + 1))
                x0_pos[:n_slopes] = 0.01
                x0_pos[n_slopes] = 5e-8
                
                x0_neg = np.zeros((n_slopes + 1))
                x0_neg[:n_slopes] = 0.01
                x0_neg[n_slopes] = 5e-8

                # Constraint that the sum of exponentials (except noise) should be greater than 0 
                def cons1(x):
                    return np.dot(envelopes[:, :-1], x[:n_slopes])
                ineq_cons = {'type': 'ineq', 'fun' : cons1}

                loss_weighting = np.ones_like(original_env_ds)
                # loss_weighting[:] = 2 

                def F(x):
                    weighted_exponentials = np.dot(envelopes, x)
                
                    return np.sum( ( (np.sqrt(original_env_ds) - np.sqrt(weighted_exponentials)) * loss_weighting )**2)
                
                pos_bounds = [(0,10) for b in range(n_slopes)]
                neg_bounds = [(-10,10) for b in range(n_slopes)]
                pos_bounds.append((0,1e-3))
                neg_bounds.append((0,1e-3))

                assert len(x0_neg) == len(neg_bounds) == len(x0_pos) == len(pos_bounds) ==  n_slopes + 1 
                # print (x0_pos, x0_neg)
                # print (pos_bounds, neg_bounds)

                res_pos = scipy.optimize.minimize(F, x0_pos, method='SLSQP', bounds=pos_bounds, tol=1e-12,  options={"disp":False, "maxiter": 5000})
                res_neg = scipy.optimize.minimize(F, x0_neg, method='SLSQP', bounds=neg_bounds, tol=1e-12, constraints=ineq_cons, options={"disp":False, "maxiter": 5000})
                lets_plot = False
                if not res_pos.success :
                    print ("pos fitting failed")
                    print (res_pos)
                    print ("RIR index", rIdx)
                    print ("Octave", config['f_bands'][bIdx])
                
                if not res_neg.success : 
                    print ("neg fitting failed")
                    print (res_neg)
                    print ("RIR index", rIdx)
                    print ("Octave", config['f_bands'][bIdx])
             
           
                    

                neg_fit_result[rIdx][cIdx][bIdx] = res_neg.x * octave_filtered_rir_norm_factor
                pos_fit_result[rIdx][cIdx][bIdx] = res_pos.x * octave_filtered_rir_norm_factor
        
                if plot :
                    counter = 0 
                    # Negative result 
                    if counter < 2 : 
                        counter += 1 
                        weighted_envelopes = envelopes_fullLength[:, :-1]  * res_neg.x[:-1]
                        total_envelopes = np.sum(weighted_envelopes, 1)
                        
                        weighted_envelopes_ds = envelopes[:, :-1]  * res_neg.x[:-1]
                        total_envelopes_ds = np.sum(weighted_envelopes_ds, 1) 
                    
                        noise = np.random.randn(L*2)
                        noise_rms = np.sqrt(np.mean(noise**2))
                        octave_filtered_noise = filterbank(torch.FloatTensor(noise))
                        
                        octave_filtered_noise = octave_filtered_noise.numpy()[0]
                        octave_filtered_noise  = octave_filtered_noise[L//2 : -L//2]
                        neg_shaped_noise = total_envelopes * octave_filtered_noise
                        neg_shaped_noise_rms = np.sqrt(np.mean(neg_shaped_noise**2)) 
                        neg_shaped_noise *= original_rms / neg_shaped_noise_rms 
                
                        # Neg env recomputed from noise 
                        neg_env = get_envelope(neg_shaped_noise, window_size)
                        neg_env_ds = neg_env[ds_start_index:] 
                    
                        # neg_modeled_noise += neg_shaped_noise
                    
                        fittedEDF_neg = np.flipud(np.cumsum(np.flipud(neg_shaped_noise**2)))
                        # Normalize to 1
                        norm_val = np.max(fittedEDF_neg, axis=-1)
                        fittedEDF_neg = fittedEDF_neg / norm_val
                        fittedEDF_neg_db = 10 * np.log10(fittedEDF_neg)

                        fig = plt.figure(figsize=(9, 6))
                
                        ax1 = plt.subplot(221)
                        ax1.plot(original_env_ds, label='original env')
                        ax1.plot(total_envelopes_ds, label='neg modeled env')
                        # plt.plot(neg_env_ds, label='neg modeled env')
                        # ax1.plot(pos_env_ds, label='pos modeled env')
                        ax1.legend()
                        # plt.show()
                    
                        
                        ax2 = plt.subplot(222)
                        ax2.plot(10 *np.log10(edf), label='original EDF')
                        ax2.plot(fittedEDF_neg_db, label='neg modeled EDF')
                        # ax2.plot(fittedEDF_pos_db_plot, label='pos modeled EDF') 
                        ax2.legend()
            
                        ax3 = plt.subplot(223)
                        ax3.plot(octave_filtered_rir, label='original rir')
                        ax3.plot(neg_shaped_noise, label='neg modeled rir')
                        ax3.legend()
                    
                        ax4 = plt.subplot(224, sharey=ax3)
                        ax4.plot(octave_filtered_rir, label='original rir')
                        # ax4.plot(pos_shaped_noise, label='pos modeled rir')
                        ax4.legend()
                        # fig.suptitle(f"{dataset_name}, RIR {rir_number}, {fBands[bIdx]} Hz")
                        plt.tight_layout()
                        plt.show()
                       
        

    if not multichannel : 
        pos_fit_result = pos_fit_result.squeeze(1)
        neg_fit_result = neg_fit_result.squeeze(1) 
        all_original_env = all_original_env.squeeze(1)

    return pos_fit_result, neg_fit_result, all_original_env



