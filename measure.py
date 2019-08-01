#!/usr/bin/env python
# coding: utf-8

# In[73]:


import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage import io
from skimage.filters import threshold_mean

from sklearn.metrics import jaccard_score

import math
import argparse

# In[74]:
parser = argparse.ArgumentParser(description='Evaluation of the prediction with the following metrics: SSIM, Jaccard index, signal-to-noise ratio. The data folder contains 4 folders with tif files of the same name: noisy(input data), clean (ground truth), to_predict and predicted.')
parser.add_argument('data_path', type=str,help='Path to your input data: noisy (low intensity), clean(high intensity) and predicted folder with .tif files')


# In[87]:


def SNR(im_ref,im_test):
    sum_ref = (im_ref**2).sum()
    diff = ((im_ref-im_test)**2).sum()
    snr = (10*math.log10(sum_ref/diff))
    return snr

def max_thresholding(im_ref,im_test):
    max_ref = np.max(im_ref,axis=0)
    max_test = np.max(im_test,axis=0)
    thresh_ref = threshold_mean(max_ref)
    binary_ref = max_ref > thresh_ref
    thresh_test = threshold_mean(max_test)
    binary_test = max_test > thresh_test
    return binary_ref, binary_test

def jaccard_measure(im_ref, im_test):
    binary_ref, binary_test = max_thresholding(im_ref,im_test)
    jacc_sc = jaccard_score(binary_test,binary_ref,average='weighted')
    return jacc_sc

def ssim_f(im_ref,im_test):
    binary_ref, binary_test = max_thresholding(im_ref,im_test)
    ssim_in = ssim(binary_ref,binary_test,multichannel=True)
    return ssim_in

def bar_plot(data,title):
    #plt.figure(figsize=(20,20))
    plt.title(title)
    medianprops = {'color': 'magenta', 'linewidth': 2}
    flierprops = {'color': 'black', 'marker': 'x'}
    plt.boxplot(data,medianprops=medianprops, flierprops = flierprops)
    plt.xticks([1,2],['noisy/GT','result/GT'])
    plt.show()

def save_csv(data,title):
    with open('metrics.csv', 'w') as f:
    	for i in range(len(data)):
        	f.write("%s,%s\n"%(title[i],data[i]))

def main():
    
    JACC_in = []
    JACC_out = []
    SSIM_in=[]
    SSIM_out = [] 
    SNR_in = []
    SNR_out = []
    
    args = parser.parse_args()
    data_path = args.data_path
    path_in = data_path+'noisy/'
    path_out = data_path+'predicted/'
    path_target = data_path+'clean/'
    
    for file_ in sorted(os.listdir(path_in)):
        
        im_target = io.imread(path_target+file_)
        im_input = io.imread(path_in+file_)
        im_output = io.imread(path_out+file_)
        
        #computing ssim and storing in separated lists for input and result
        ssim_in = ssim_f(im_target,im_input)
        SSIM_in.append(ssim_in)
        ssim_out = ssim_f(im_target,im_output)
        SSIM_out.append(ssim_out)
        data_ssim = [SSIM_in,SSIM_out]

        
        #computing jaccard index and storing in separated lists for input and result
        jacc_in = jaccard_measure(im_target, im_input)
        JACC_in.append(jacc_in)
        jacc_out = jaccard_measure(im_target, im_output)
        JACC_out.append(jacc_out)
        data_jacc = [JACC_in,JACC_out]

        #computing SNR and storing in separated lists for input and result
        snr_in = SNR(im_target, im_input)
        SNR_in.append(snr_in)
        snr_out = SNR(im_target, im_output)
        SNR_out.append(snr_out)
        data_snr = [SNR_in,SNR_out]

    #bar_plot(data_snr,data_jacc,data_ssim)
    DATA = [data_snr,data_jacc,data_ssim]
    num_metrics = len(DATA)
    #fig,ax = plt.subplot()
    #plt.subplot(1,3)
    fig=plt.figure()
    for count,value in enumerate(DATA):
        title =['Signal to Noise Ratio','Jaccard index','Structural similarity index']
        fig.add_subplot(1,num_metrics,count+1)
        bar_plot(DATA[count],title[count])
    save_csv(DATA,title)
if __name__ == '__main__':
    main()

