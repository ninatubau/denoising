""" Prediction for image denoising using a pre-trained model. 

This python code demonstrates applying a CARE model for a 3D denoising task, assuming that training was already completed via training.py.  
The trained model is assumed to be located in the folder 'models' with the name given by the user (default `my_model`).
 
This script requires 'os','sys','numpy','matplotlib','tifffile' and csbdeep modules to be imported.

	* parser_init - initialises the script parameters
	* predict - predicts the denoised image thanks to a pre-trained model
	* reconstruction - reconstruct the image calling predict function
	* pot_results - plots the output of the network for visualisation  
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
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
import pathlib


parser = argparse.ArgumentParser(description='Prediction arguments')
parser.add_argument('path_data', type=str,help='Path to your input data to predict')
parser.add_argument('model_name', type=str,help='Name of the model previously trained with training.py')
parser.add_argument('--n_tiles', nargs="+",type=int, default=(1,4,4),help='Tuple of the number of tiles for every image axis to avoid out of memory problems when the input image is too large. Examples: 1 4 4')
parser.add_argument('--axes', type=str,default='XYZ',help='Axes to indicate the semantic order of the images axes. Examples : ZYX, CXY ... ')
parser.add_argument('--plot_prediction', type=bool,default =False,help='Plotting images of the prediction : True or False')
parser.add_argument('--filter_data', type=str, default='all', help ='Filter the data that you want to predict: ch0,ch1, all')
parser.add_argument('--stack_nb', type=int, default=10, help ='Number of images taking as input')

def parser_init(parser):
	"""Initialises the arguments of the script with argparse

	Parameters
	----------
	parser : argparse
		The input arguments to run the script
	
	Returns
	-------
	path_data 
	model_name
	n_tiles
	axes
	plot_prediction
	filter_data
	stack_nb
	"""	
	args = parser.parse_args()
	path_data = args.path_data
	model_name = args.model_name
	n_tiles=tuple(args.n_tiles)
	axes = args.axes
	plot_prediction = args.plot_prediction
	filter_data = args.filter_data
	stack_nb= args.stack_nb
	return path_data, model_name, n_tiles, axes, plot_prediction, filter_data, stack_nb

def predict(path_data, model_name, n_tiles, axes, plot_prediction,stack_nb, filter_data):
	"""Predicts the output of the netowrk using a pre-trained model.
	
	Parameters
	----------
	path_data : str
		Path to input data to predict
	model_name : str	
		Name of the pre-trained model 
	n_tiles : tuple
		Size of tile to split the patches (helps avoidint out of memory problems when predicting
	axes : str
		Semantic order of the channels
	plot_prediction : bool
		True or False whether the prediction is plot or not
	"""
	model = CARE(config=None, name=model_name, basedir='models')


	for file_ in sorted(os.listdir(path_data)):

		if file_.endswith('.tif') and not pathlib.Path(os.path.dirname(os.getcwd())+'/predicted/'+file_).exists() :

			if filter_data in file_ : 
				reconstruction(model, file_,path_data,axes,n_tiles, plot_prediction)

			elif filter_data=='all':
				reconstruction(model, file_,path_data,axes,n_tiles, plot_prediction)


def reconstruction(model, file_name, path_data, axes,n_tiles, plot_prediction):
	"""Reconstruct the whole image and saves it in a 'predicted' folder. 

	Parameters
	----------
	model : object
		Model used to predict the data
	file_name : str
		Name of the file to predict
	path_data : str
		Path where the data to predict is saved
	axes : str
		Semantic order of the channels
	plot prediction : bool
		True or False whether the prediction is plot or not
		
	"""
	print('Reading file: ',file_name)
	start =time.time()
	x = imread(path_data+'/'+file_name)
	#n_tiles to avoid *Out of memory* problems during `model.predict`
	print(x.shape)
	#res=[]
	#for i in range(x.shape[0]):
	restored=model.predict(x,axes,n_tiles=n_tiles)
	#res.append(restored)
	#restored = np.stack(res,axis=0)
			
	end = time.time()
	print('Prediction time %s sec ' %(end - start))
	print('Saving file: ',file_name)
	os.chdir(path_data)
	os.chdir('..')
	if not os.path.exists('predicted'):
		os.makedirs('predicted')
	imsave(os.getcwd()+'/predicted/'+file_name, restored)
	

def plot_results(x,restored):
	"""Max project and plot some of the predicted images.
	
	Parameters
	----------
	x : np.array
		Array containing the noisy image (before the network).
	restored : np.array
		Array containint the same image after the network (denoised image).
	"""
	plt.figure(figsize=(20,20))
	plt.subplot(121)
	plt.imshow(np.max(x,axis=0),cmap='gray')
	plt.title('Maximum projection input image (before CARE)')
	plt.subplot(122)
	plt.imshow(np.max(restored,axis=0),cmap='gray')
	plt.title('Maximum projection restored image (after CARE)')
	plt.show()
		    
if __name__ == '__main__':
	path_data, model_name, n_tiles, axes, plot_prediction, filter_data,stack_nb = parser_init(parser)
	predict(path_data, model_name, n_tiles,axes, plot_prediction, stack_nb, filter_data)
	if plot_prediction:
		plot_results(x, restored)

