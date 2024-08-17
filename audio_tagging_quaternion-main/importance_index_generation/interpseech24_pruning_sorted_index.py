#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:17:21 2024

@author: arshdeep
"""

#%% import modules


import sys
sys.path.insert(0,'~/audio_tagging_quaternion-main/pytorch')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from collections import OrderedDict
# import os
# os.chdir('/home/arshdeep/PANNs_code/audioset_tagging_cnn-master/pytorch')
from pytorch_utils import do_mixup, interpolate, pad_framewise_output
# import torch, torchvision
from torch.utils.data import Dataset
# from torchvision import datasets
from torchvision.transforms import ToTensor
from quaternion_layers import QuaternionConv2d ,QuaternionConv2dNoBias, QuaternionConv, QuaternionLinear
from torchinfo import summary
import os
import numpy as np    
from codecarbon import EmissionsTracker

import torchaudio

from scipy.stats.mstats import gmean
from scipy.spatial import distance
import matplotlib.pyplot as plt
from time import process_time
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


#%% load weights

checkpoint_path =  '~/audio_tagging_quaternion-main/checkpoint/980000_iterations.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) #model.state_dict()#
# model.load_state_dict(torch.load(checkpoint_path))
# model.eval()
# print(model)
# print(checkpoint)

weights = checkpoint['model'] #['model']
weights_pruned = weights #checkpoint['model'] #['model']



#%% importance score generation frameworks

def rank1_apporx(data):
    u,w,v= np.linalg.svd(data)
    M = np.matmul(np.reshape(u[:,0],(-1,1)),np.reshape(v[0,:],(1,-1)))
    M_prototype = M[:,0]/np.linalg.norm(M[:,0],2)
    return M_prototype


#% entry-wise l_1 norm based scores
def QCNN_L1_Imp_index(W):
	Score=[]
	for Nf in range(np.shape(W)[0]):
		Score.append(np.sum(np.abs(W[Nf,:,0])))
	return Score/np.max(Score)



#%% impoarant score calculation for each channel separately

path = '~/audio_tagging_quaternion-main/sorted_index_L1_pruning/'



conv_key_list = ['conv_block6.conv2.r_weight','conv_block6.conv2.k_weight','conv_block6.conv2.j_weight','conv_block6.conv2.i_weight','conv_block6.conv1.r_weight','conv_block6.conv1.k_weight', 'conv_block6.conv1.j_weight',
'conv_block6.conv1.i_weight',
'conv_block5.conv2.r_weight',
'conv_block5.conv2.k_weight',
'conv_block5.conv2.j_weight',
'conv_block5.conv2.i_weight',
'conv_block5.conv1.r_weight',
'conv_block5.conv1.k_weight',
'conv_block5.conv1.j_weight',
'conv_block5.conv1.i_weight',
'conv_block4.conv2.r_weight',
'conv_block4.conv2.k_weight',
'conv_block4.conv2.j_weight',
'conv_block4.conv2.i_weight',
'conv_block4.conv1.r_weight',
'conv_block4.conv1.k_weight',
'conv_block4.conv1.j_weight',
'conv_block4.conv1.i_weight',
'conv_block3.conv2.r_weight',
'conv_block3.conv2.k_weight',
'conv_block3.conv2.j_weight',
'conv_block3.conv2.i_weight',
'conv_block3.conv1.r_weight',
'conv_block3.conv1.k_weight',
'conv_block3.conv1.j_weight',
'conv_block3.conv1.i_weight',
'conv_block2.conv2.r_weight',
'conv_block2.conv2.k_weight',
'conv_block2.conv2.j_weight',
'conv_block2.conv2.i_weight',
'conv_block2.conv1.r_weight',
'conv_block2.conv1.k_weight',
'conv_block2.conv1.j_weight',
'conv_block2.conv1.i_weight',
'conv_block1.conv2.r_weight',
'conv_block1.conv2.k_weight',
'conv_block1.conv2.j_weight',
'conv_block1.conv2.i_weight',
'conv_block1.conv1.r_weight',
'conv_block1.conv1.k_weight',
'conv_block1.conv1.j_weight',
'conv_block1.conv1.i_weight']




for key in conv_key_list:
    W_2D = weights[key].numpy()
    W = np.reshape(W_2D,(np.shape(W_2D)[0],np.shape(W_2D)[1],np.shape(W_2D)[2]*np.shape(W_2D)[3]))
    print(W.shape)  # W Shape: [FILTERS, CHANNELS, HEIGHT, WIDTH]
    score = QCNN_L1_Imp_index(W)
    sorted_index = np.argsort(score)
    # weights[key] = torch.tensor(W_2D[0:50,:,:,:])
    filename = path + key + '.npy'
    np.save(filename, np.array(sorted_index).astype(int))
    print(len(sorted_index),key)
    
    
    


    
#%% avergae score across each channel...









    
