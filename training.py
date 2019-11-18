# Demo: Neural network training for denoising of *Tribolium castaneum*
# 
# This python code demonstrates training a CARE model for a 3D denoising task calling the datagen code that saves to disk to the file ``data/data_preparednpz``.
# 
# Note that training a neural network for actual use should be done on more (representative) data and with more training time.
# 
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.



from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys

#file_dir = os.path.dirname('./../../')
#sys.path.append(file_dir)

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

from datagen import data_generation 
import argparse 
import gc

parser = argparse.ArgumentParser(description='Data generation and training. The data folder contains 4 folders with tif files of the same name: noisy(input data), clean (ground truth), to_predict and predicted.')
parser.add_argument('data_path', type=str,help='Path to your input data: noisy (low intensity) and clean(high intensity) folder with .tif files')
parser.add_argument('--axes', type=str, default='XYZ',help='Axes to indicate the semantic order of the images axes. Examples : ZYX, CXY ... ')
parser.add_argument('--validation_split', type=int,default =0.1,help='Ratio of validation data for training')
parser.add_argument('--train_steps_per_epochs', type=int,default =100,help='Number of training steps per epochs')
parser.add_argument('--train_epochs', type=int,default =10,help='Number of epochs')
parser.add_argument('--model_name', type=str,default ='my_model',help='Name of the model to save')


def parser_init(parser):
	
	args = parser.parse_args()
	path_data = args.data_path
	axes = args.axes
	validation_split = args.validation_split
	train_steps_per_epochs =args.train_steps_per_epochs
	train_epochs = args.train_epochs
	model_name = args.model_name

	return path_data, axes, validation_split, train_steps_per_epochs, train_epochs, model_name

def load(path_data,axes,validation_split):
	# The TensorFlow backend uses all available GPU memory by default, hence it can be useful to limit it:
	limit_gpu_memory(fraction=1/2)
	#call datagen to generate the data properly and save to `data/data_prepared.npz``
	data_generation(path_data,axes)

	# Load training data generated via [datagen.py, use 10% as validation data by default
	(X,Y), (X_val,Y_val), axes = load_training_data('data/data_prepared.npz', validation_split, verbose=True)

	return (X,Y), (X_val,Y_val)

	# # CARE model
	# 
	# Before we construct the actual CARE model, we have to define its configuration via a `Config` object, which includes 
	# * parameters of the underlying neural network,
	# * the learning rate,
	# * the number of parameter updates per epoch,
	# * the loss function, and
	# * whether the model is probabilistic or not.

def train(X,Y,X_val,Y_val,axes,train_steps_per_epochs,train_epochs,model_name):	

	config = Config(axes, n_channel_in=1, n_channel_out=1, train_steps_per_epoch=train_steps_per_epochs,train_epochs=train_epochs)

	# We now create a CARE model with the chosen configuration:
	model = CARE(config, model_name, basedir='models')



	# # Training
	# 
	# Training the model will likely take some time. We recommend to monitor the progress with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) (example below), which allows you to inspect the losses during training.
	
	# You can start TensorBoard from the current working directory with `tensorboard --logdir=.`
	# Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.

	history = model.train(X,Y, validation_data=(X_val,Y_val))
	model.export_TF()
	return history

def save_results(history):
	plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])
	
	#save model results in loss.csv file
	with open('loss.csv', 'w') as f:
    		for key in history.history.keys():
        		f.write("%s,%s\n"%(key,history.history[key]))

	# # Export model to be used with CSBDeep **Fiji** plugins and **KNIME** workflow
	gc.collect()

if __name__ == '__main__':
	path_data, axes, validation_split, train_steps_per_epochs, train_epochs, model_name = parser_init(parser)
	(X,Y), (X_val,Y_val) = load(path_data,axes,validation_split)
	history = train(X,Y,X_val,Y_val,axes,train_steps_per_epochs,train_epochs,model_name)
	save_results(history)

