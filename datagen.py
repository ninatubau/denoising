#!/usr/bin/env python
# coding: utf-8

# 
# Demo: Training data generation for denoising of *Tribolium castaneum*
# 
# This python code demonstrates training data generation for a 3D denoising task, where corresponding pairs of low and high quality stacks can be acquired. 
# 
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.
from __future__ import print_function, unicode_literals, absolute_import, division

import os
import sys

#place system to have access to csbdeep files
file_dir = os.path.dirname('./../../')
sys.path.append(file_dir)


import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#import imp
#sys.path.remove('/stornext/Img/data/prkfs1/m/Microscopy/Nina_Tubau/Image_reconstruction/n2v')



# # Generate training data for CARE
# 
# We first need to create a `RawData` object, which defines how to get the pairs of low/high SNR stacks and the semantics of each axis (e.g. which one is considered a color channel, etc.).
# 
# Here we have two folders "low" and "GT", where corresponding low and high-SNR stacks are TIFF images with identical filenames.  
# For this case, we can simply use `RawData.from_folder` and set `axes = 'ZYX'` to indicate the semantic order of the image axes. 
def data_generation(data_path,axes,patch_size):
	print('Generating .npz file with the data')
	#dir_data = './../../../../DATA/data_lattice/stack_2/GPUdecon/'
	#ax = 'ZYX'
	raw_data = RawData.from_folder (
	    basepath    = data_path,
	    source_dirs = ['noisy'],
	    target_dir  = 'clean',
	    axes        = axes,
	)


# From corresponding stacks, we now generate some 3D patches. As a general rule, use a patch size that is a power of two along XYZT, or at least divisible by 8.  
# Typically, you should use more patches the more trainings stacks you have. By default, patches are sampled from non-background regions (i.e. that are above a relative threshold), see the documentation of `create_patches` for details.
# 
# Note that returned values `(X, Y, XY_axes)` by `create_patches` are not to be confused with the image axes X and Y.  
# By convention, the variable name `X` (or `x`) refers to an input variable for a machine learning model, whereas `Y` (or `y`) indicates an output variable.



	X, Y, XY_axes = create_patches (
	    raw_data            = raw_data,
	    patch_size          = patch_size,
	    n_patches_per_image = 1024,
	    save_file           = 'data/data_prepared.npz',
	)


	#verification of dimensions
	assert X.shape == Y.shape
	print("shape of X,Y =", X.shape)
	print("axes  of X,Y =", XY_axes)



	# This shows the maximum projection of some of the generated patch pairs (even rows: *source*, odd rows: *target*)

	for i in range(2):
	    plt.figure(figsize=(16,4))
	    sl = slice(8*i, 8*(i+1)), 0
	    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
	    plt.show()
	None;

