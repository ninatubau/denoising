
#!/usr/bin/env python
# coding: utf-8

# This python code demonstrates applying a CARE model for a 3D denoising task, assuming that training was already completed via training.py.  
# The trained model is assumed to be located in the folder `models` with the name given by the user (default `my_model`).
# 
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys

file_dir = os.path.dirname('./../../')
sys.path.append(file_dir)


import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE
from tifffile import imsave
import time
import argparse 

parser = argparse.ArgumentParser(description='Prediction arguments')
parser.add_argument('path_data', type=str,help='Path to your input data: read the to_predict image')
parser.add_argument('name_model', type=str,default ='my_model',help='Name of the model to use')
parser.add_argument('--n_tiles', nargs="+",type=int, default=(1,4,4),help='Tuple of the number of tiles for every image axis to avoid out of memory problems when the input image is too large. Examples: 1 4 4')
parser.add_argument('--axes', type=str,default='ZYX',help='Axes to indicate the semantic order of the images axes. Examples : ZYX, CXY ... ')
parser.add_argument('--plot_prediction', type=bool,default =False,help='Plotting images of the prediction : True or False')

def main():
	
	args = parser.parse_args()
	path_data = args.path_data
	name_model = args.name_model
	n_tiles=tuple(args.n_tiles)
	axes = args.axes
	plot_prediction = args.plot_prediction
	# # CARE model - our data

	# Load trained model (located in base directory `models` with name `my_model`) from disk.  
	# The configuration was saved during training and is automatically loaded when `CARE` is initialized with `config=None`.

	model = CARE(config=None, name=name_model, basedir='models')


	# ## Apply CARE network to raw image

	for file_ in sorted(os.listdir(path_data+'to_predict/')):
	    #if (file.startswith('1')):
		print('Reading file: ',file_)
		start =time.time()
		x = imread(path_data+'to_predict/'+file_)
        #x = np.expand_dims(x,axis=3)
		#mid = int(x.shape[0]/2)
		#restored_1 = model.predict(x[0:mid], axes)
		#restored_2 = model.predict(x[mid:],axes)
		#restored = np.concatenate((restored_1,restored_2),axis=0)
		restored=model.predict(x,axes,n_tiles=n_tiles)
		end = time.time()
		print('Prediction time %s sec ' %(end - start))
		print('Saving file: ',file_)
		imsave(path_data+'predicted/'+file_, restored)


	# Alternatively, one can directly set `n_tiles` to avoid the time overhead from multiple retries in case of memory issues.
	# 
	# **Note**: *Out of memory* problems during `model.predict` can also indicate that the GPU is used by another process. In particular, shut down the training notebook before running the prediction (you may need to restart this notebook).


	# # Raw low/high-SNR image and denoised image via CARE network
	# 
	# Plot the test stack pair and the predicted restored stack (middle).



	#plot the results
	if (plot_prediction):
		plt.figure(figsize=(20,20))
		plt.subplot(121)
		plt.imshow(np.max(x,axis=0),cmap='gray')
		plt.title('Maximum projection input image (before CARE)')
		plt.subplot(122)
		plt.imshow(np.max(restored,axis=0),cmap='gray')
		plt.title('Maximum projection restored image (after CARE)')
		plt.show()
		    
if __name__ == '__main__':
	main()

