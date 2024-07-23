import os
import numpy as np 
from pathlib import Path 
import mat73
import soundfile as sf 
import yaml
import pickle 
import matplotlib.pyplot as plt 
config = yaml.safe_load(open(os.path.abspath("/Users/kyungyunlee/dev/fade-in-reverb/fade_in_reverb/config.yaml")))


def load_measurement_dataset() : 
    # Load RIR 
    rirs_tmp = []
    for pos in range(80)  :
        if pos in [26,27,28] : 
            continue 
        rir_files = list(Path(os.path.abspath("/Users/kyungyunlee/dev/fade-in-reverb/data/rirs_med")).glob(f"hallways-lecturehall*{pos:02d}*zoom*.wav"))
        curr_pos_rir = np.zeros((96000, 4)) 
        for rir_file in rir_files : 
            y, fs = sf.read(rir_file)
            print (y.shape)
            curr_pos_rir += y
        for i in range(4) : 
            rirs_tmp.append(curr_pos_rir[:, i]) 
    rirs_tmp = np.array(rirs_tmp)

    rirs_tmp = rirs_tmp + np.random.randn(rirs_tmp.shape[0], rirs_tmp.shape[1]) * 1e-8
    # Preprocess 
    rirs = [] 
    mask=np.ones(10)/10

 
    for i in range(len(rirs_tmp)) : 
        curr_rir = rirs_tmp[i]
        log_energy = 10 * np.log10(np.convolve(curr_rir**2, mask))
        noise_floor_level = np.mean(log_energy[500 : 1500]) 
        direct_thresh = noise_floor_level + 20
        noise_thresh = noise_floor_level 
        
        front_index = np.where(log_energy[:8000] > direct_thresh)
        # print (front_index)
        if len(front_index[0]) == 0 :        

            # just remove noise 
            front_index = np.where(log_energy[:8000] > noise_thresh)[0][0]
            front_index = max(0, front_index - 200)
            cut_rir = curr_rir[front_index :]
            

        else : 
            front_index = front_index[0][0]
       
            front_index = max(0, front_index - 200)
            cut_rir = curr_rir[front_index :]


        if len(cut_rir) < 96000: 
            # Pad
            tmp = np.zeros((96000,))
            tmp[:len(cut_rir)] = cut_rir
            cut_rir =tmp 
        else :
            cut_rir = cut_rir[:96000]


        rirs.append(cut_rir) 
    
    
    rirs = np.array(rirs)


    # Normalization factor 
    rir_norm_factor = np.max(np.abs(rirs))


    rirs = rirs / rir_norm_factor



    # Load position info 
    
    rir_df = pickle.load(open(os.path.abspath("/Users/kyungyunlee/dev/fade-in-reverb/data/blind_measurement/all_scenes_with_position_data.pkl"), "rb"))
    scene_df = rir_df[rir_df['scene'] == 'hallways-lecturehall']
    
    trajectory = list(zip(list(scene_df['posX']),list(scene_df['posY']), list(scene_df['posZ'])))
    
    trajectory = np.array(trajectory)
    

    assert len(rirs) == len(trajectory) * 4 

    return rirs, trajectory


def load_simulation_dataset():
    # MAT 
    data_dict = mat73.loadmat(os.path.abspath('/Users/kyungyunlee/dev/fade-in-reverb/data/treble_3room/three_coupled_rooms_4s_fine_normal_calibrated_srirs.mat'))
    rirs = data_dict['srirDataset']['srirs']


    # Use full 3 second RIR for computing common decay times 
    rirs = rirs[:, :128000, :]
    # Use first 2 second RIR for performing the fitting 
    omni_rirs = rirs[:, :96000, 0]
    
    nRIRs, L, nChannels = rirs.shape
    rirs = np.transpose(rirs, (0, 2, 1))
    rirs = rirs[:, 0:1, :]
    total_rirs = np.reshape(rirs, (nRIRs*1, L))

    # total_rirs = total_rirs + np.random.randn(total_rirs.shape[0], total_rirs.shape[1])*2e-8
    # omni_rirs = omni_rirs + np.random.randn(omni_rirs.shape[0], omni_rirs.shape[1])*2e-8
    
    # Preprocess omni RIRs only 
    # omni_rirs_processed = [] 
    # mask=np.ones(10)/10


    # direct_thresh = -30

    # noise_thresh = -70

    # for i in range(0, len(omni_rirs)) : 
    #     curr_rir = omni_rirs[i]
    #     log_energy = 10 * np.log10(np.convolve(curr_rir**2, mask))


    #     front_index = np.where(log_energy[:700] > direct_thresh)
    #     if len(front_index[0]) > 0 : 
    #         direct_loc = front_index[0][0] - 20
    #         cut_rir = curr_rir[direct_loc:]
            
    #     else : 
    #         # remove noise 
    #         front_index = np.where(log_energy[:2000] > noise_thresh)
    #         if len(front_index[0]) > 0 : 
    #             start_loc = front_index[0][0]
    #         else :
    #             start_loc = 2000 
            
    #         start_loc = start_loc - 200
    #         cut_rir = curr_rir[start_loc:]


    #     if len(cut_rir) < 96000: 
    #         # Pad
    #         tmp = np.zeros((96000,))
    #         tmp[:len(cut_rir)] = cut_rir
    #         cut_rir =tmp 
    #     else :
    #         cut_rir = cut_rir[:96000]

    #     omni_rirs_processed.append(cut_rir)
    # omni_rirs_processed = np.array(omni_rirs_processed) 


    # Load position data 
    pos_data = np.load(os.path.abspath("/Users/kyungyunlee/dev/fade-in-reverb/data/treble_3room/position_data.npz"))
    rcvPos = pos_data['rcvPos']
    srcPos = pos_data['srcPos']

    # # Process data 
    # rirs = rirs + np.random.randn(rirs.shape[0], rirs.shape[1])*2e-8
    # rirs = rirs[:, :96000]


    return total_rirs, omni_rirs, rcvPos, srcPos 

