# {
#    "owner": "kdevasenapathy",
#    "repository": "cs159skurves.github.io",
#    "path": "kdevasen/C5/c345Us1efg/cs159skurves.github.io/COLGE/",
#    "tokenInfo": "{{ssm-secure:SecureString_parameter_name}}"
# }
# Execution Timeout: 72000
# Write command output to an S3 bucket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import time
start_time = time.time()

batch_size = 128
nodes = 20
graph_nbr = 100

def moving_average(np_arr, window=1):
    return np.convolve(np_arr, np.ones(window), mode='valid')/window

def read_output_file(folder_name='', out_type='final', epoch=0, mode='absolute', avg_window_size=1):
    current_path = pathlib.Path().absolute()
    directory_path = current_path / folder_name
    
    if out_type == 'opt':
        filename = 'opt_set.out'
    elif out_type == 'final':
        filename = 'test.out'
    elif out_type == 'epoch':
        filename = 'test_'
        if mode == 'approx':
            filename += 'approx_'
        filename += str(epoch) + '.out'
    else:
        print("Unrecognized output type")
        return None
        
    file_path = directory_path / filename
    return moving_average(np.loadtxt(file_path, dtype=np.float32), avg_window_size)


# Function to figure out which epoch had the lowest convergence ratio
def best_epoch(folder_name, epochs = 0):
    min_convergence = 0
    min_epoch = -1
    for i in range(epochs):
        candidate = np.min(read_output_file(folder_name, out_type='epoch', i, mode='absolute', avg_window_size=100))
        if min_epoch == -1:
            min_convergence = candidate
            min_epoch = i
        elif cnadidate < min_convergence:
            min_convergence = candidate
            min_epoch = i
    return epoch, min_conv


# Max independent set set
best_nstep = 5
best_lrate = 1e-4
best_g = 0.99999
best_decay = 0.75
max_epochs = 15

nstep_candidates = [4, 5, 6]
min_conv = 1000
for nstep in nstep_candidates:
    filepath = ("runs/MaxIndependent/optimizer_n_step" + str(nstep))
    start_time = time.time()
    %run main.py --model S2V_QN_1 --path filepath --environment_name MIS --graph_type barabasi_albert --m 0.4 --node nodes --ngames 100 --n_step nstep --lr best_lrate --gamma best_g --lr_decay best_decay --bs batch_size --epoch max_epochs
    end_time = time.time()
    print("max independent nstep = ", nstep, "time = ", end_time - start_time)
    epoch, min_conv_cand = best_epoch(filepath, max_epochs)
    if min_conv_cand <= min_conv:
        best_nstep = nstep



lrate_candidates = [1e-3, 1e-4, 1e-5, 1e-6]
decay_candidates = [0.65, 0.7, 0.75, 0.8, 0.85]

gridsearch_params = [
        (lrate, decay)
        for lrate in lrate_candidates
        for decay in decay_candidates
    ]

min_conv = 1000
for lrate, decay in gridsearch_params:
    filepath = ("runs/MaxIndependent/optimizer_lrate_" + str(lrate) + "_decay_" + str(decay))
    start_time = time.time()
    %run main.py --model S2V_QN_1 --path filepath --environment_name MIS --graph_type barabasi_albert --m 0.4 --node nodes --ngames 100 --n_step best_nstep --lr lrate --gamma best_g --lr_decay decay --bs batch_size --epoch max_epochs
    end_time = time.time()
    print("max independent lrate = ", lrate, ", decay = ", decay, "time = ", end_time - start_time)
    epoch, min_conv_cand = best_epoch(filepath, max_epochs)
    if min_conv_cand <= min_conv:
        best_lrate = lrate
        best_decay = decay

g_candidates = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.9999]
min_conv = 1000
for g in g_candidates:
    filepath = ("runs/MaxIndependent/optimizer_g_" + str(g))
    start_time = time.time()
    %run main.py --model S2V_QN_1 --path filepath --environment_name MIS --graph_type barabasi_albert --m 0.4 --node nodes --ngames 100 --n_step best_nstep --lr best_lrate --gamma g --lr_decay best_decay --bs batch_size --epoch max_epochs
    end_time = time.time()
    print("max independent gamma = ", g, "time = ", end_time - start_time)
    epoch, min_conv_cand = best_epoch(filepath, max_epochs)
    if min_conv_cand <= min_conv:
        best_g = g
        
# Get final number of epochs
filepath = ("runs/MaxIndependent/optimizer_g_" + str(best_g))
epoch, min_conv_cand = best_epoch(filepath, max_epochs)
print("Maximum Independent Set: ")
print("Epochs: ", epoch)
print("Ratio: ", min_conv_cand)
print("N-step: ", best_nstep)
print("Lrate: ", best_lrate)
print("Lrate Decay: ", best_decay)
