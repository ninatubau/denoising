# This python code demonstrates applying a CARE model for a 3D denoising task, assuming that training was already completed via training.py.  
# The trained model is assumed to be located in the folder `models` with the name given by the user (default `my_model`).
# 
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

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
import tkinter


parser = argparse.ArgumentParser(description='Prediction arguments')
parser.add_argument('path_data', type=str,help='Path to your input data to predict')
parser.add_argument('name_model', type=str,default ='my_model',help='Name of the model to use')
parser.add_argument('--n_tiles', nargs="+",type=int, default=(1,4,4),help='Tuple of the number of tiles for every image axis to avoid out of memory problems when the input image is too large. Examples: 1 4 4')
parser.add_argument('--axes', type=str,default='XYZ',help='Axes to indicate the semantic order of the images axes. Examples : ZYX, CXY ... ')
parser.add_argument('--plot_prediction', type=bool,default =False,help='Plotting images of the prediction : True or False')
parser.add_argument('--filter_data', type=str, default='all', help ='Filter the data that you want to predict: ch0,ch1, all')
parser.add_argument('--stack_nb', type=int, default=10, help ='Filter the number of images taken as input')


def parser_init(parser):
	
	args = parser.parse_args()
	path_data = args.path_data
	name_model = args.name_model
	n_tiles=tuple(args.n_tiles)
	axes = args.axes
	plot_prediction = args.plot_prediction
	filter_data = args.filter_data
	stack_nb = args.stack_nb

	return path_data, name_model, n_tiles, axes, plot_prediction, filter_data, stack_nb
	# # CARE model - our data

	# Load trained model (located in base directory `models` with name `my_model`) from disk.  
	# The configuration was saved during training and is automatically loaded when `CARE` is initialized with `config=None`.

def callback():
	follow = True
	print(follow)

def predict(path_data, name_model, n_tiles, axes, plot_prediction,stack_nb, filter_data):

	model = CARE(config=None, name=name_model, basedir='models')
	#Apply CARE network to raw image

	for file_ in sorted(os.listdir(path_data)):
		print('Processing....', file_)
		if file_.endswith('.tif') and not pathlib.Path(os.path.dirname(os.getcwd())+'/predicted/'+file_).exists() :
			print('File end with .tif and doesn t exist')
			if filter_data in file_ : 
				reconstruction(model, file_,path_data,axes,n_tiles, plot_prediction)
				print('filter_data')

			if filter_data=='all':
				print('reconstruction')
				reconstruction(model, file_,path_data,axes,n_tiles, plot_prediction)

			#return x, restored

def reconstruction(model, file, path_data, axes,n_tiles, plot_prediction):

	print('Reading file: ',file)
	start =time.time()
	x = imread(path_data+'/'+file)
	#n_tiles to avoid *Out of memory* problems during `model.predict` 
	restored=model.predict(x,axes,n_tiles=n_tiles)

	if plot_prediction :
		print(plot_prediction)
					#plot_results(x,restored)
					#window = tkinter.Tk()
					#window.title("Accurate prediction?")
					#b = Button(window, text = 'OK', command = callback)
					#print(b)
	end = time.time()
	print('Prediction time %s sec ' %(end - start))
	print('Saving file: ',file)
	os.chdir(path_data)
	os.chdir('..')
	if not os.path.exists('predicted'):
		os.makedirs('predicted')
	imsave(os.getcwd()+'/predicted/'+file, restored)


def plot_results(x,restored):
		plt.figure(figsize=(20,20))
		plt.subplot(121)
		plt.imshow(np.max(x,axis=0),cmap='gray')
		plt.title('Maximum projection input image (before CARE)')
		plt.subplot(122)
		plt.imshow(np.max(restored,axis=0),cmap='gray')
		plt.title('Maximum projection restored image (after CARE)')
		plt.show()
		    
if __name__ == '__main__':
	path_data, name_model, n_tiles, axes, plot_prediction, filter_data, stack_nb = parser_init(parser)
	predict(path_data, name_model, n_tiles,axes, plot_prediction, stack_nb, filter_data)
	#if plot_prediction:
	#	plot_results(x, restored)

