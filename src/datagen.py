""" Reads and generated pairs of patches from images.

This python code demonstrates training data generation for a 3D denoising task, where corresponding pairs of low and high quality stacks can be acquired. 

"""

import os
import sys

#place system to have access to csbdeep files
#file_dir = os.path.dirname('./../../')
#sys.path.append(file_dir)


import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#import imp
#sys.path.remove('/stornext/Img/data/prkfs1/m/Microscopy/Nina_Tubau/Image_reconstruction/n2v')


def data_generation(data_path,axes,patch_size,data_name):
	"""Generates training data for training CARE network. `RawData` object defines how to get the pairs of low/high SNR stacks and the semantics of each axis. We have two folders "noisy" and "clean" where corresponding low and high-SNR stacks are TIFF images with identical filenames. 

	Parameters
	----------
	data_path : str
		Path of the input data containing 'noisy' and 'clean' folder
	axes : str
		Semantic order each axes
	patch_size : tuple 
		Size of the patches to crop the images
	data_name : str
		Name of the .npz file containing the pairs of images.
	
		
 
	"""
	raw_data = RawData.from_folder (
	    basepath    = data_path,
	    source_dirs = ['noisy'],
	    target_dir  = 'clean',
	    axes        = axes,
	)

	# Patch size that is a power of two along XYZT, or at least divisible by 8.  
	# By convention, the variable name `X` (or `x`) refers to an input variable for a machine learning model, whereas `Y` (or `y`) indicates an output variable.

	X, Y, XY_axes = create_patches (
	    raw_data            = raw_data,
	    patch_size          = patch_size,
	    n_patches_per_image = 1024,
	    save_file           = 'data/'+data_name+'.npz',
	)

	assert X.shape == Y.shape
	print("shape of X,Y =", X.shape)
	print("axes  of X,Y =", XY_axes)


	#for i in range(2):
	#    plt.figure(figsize=(16,4))
	#    sl = slice(8*i, 8*(i+1)), 0
	#    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
	#    plt.show()

