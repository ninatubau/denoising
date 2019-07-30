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

file_dir = os.path.dirname('./../../')
sys.path.append(file_dir)

import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

from datagen import data_generation 
import argparse 

parser = argparse.ArgumentParser(description='Data generation and training. The data folder contains 4 folders with tif files of the same name: noisy(input data), clean (ground truth), to_predict and predicted.')
parser.add_argument('data_path', type=str,help='Path to your input data: noisy (low intensity) and clean(high intensity) folder with .tif files')
parser.add_argument('--axes', type=str, default='XYZ',help='Axes to indicate the semantic order of the images axes. Examples : ZYX, CXY ... ')
parser.add_argument('--validation_split', type=int,default =0.1,help='Ratio of validation data for training')
parser.add_argument('--train_steps_per_epochs', type=int,default =100,help='Number of training steps per epochs')
parser.add_argument('--train_epochs', type=int,default =10,help='Number of epochs')
parser.add_argument('--plot_evaluation', type=bool,default =False,help='Plotting images to evaluate the result : True or False')
parser.add_argument('--model_name', type=str,default ='my_model',help='Name of the model to save')

def main():
	
	args = parser.parse_args()
	path_data = args.data_path
	axes = args.axes
	validation_split = args.validation_split
	train_steps_per_epochs =args.train_steps_per_epochs
	train_epochs = args.train_epochs
	plot_evaluation = args.plot_evaluation
	model_name = args.model_name

	# The TensorFlow backend uses all available GPU memory by default, hence it can be useful to limit it:
	limit_gpu_memory(fraction=1/2)

	#call datagen to generate the data properly and save `data/data_preparednpz``
	data_generation(path_data,axes)


	# Load training data generated via [1_datagen.ipynb](1_datagen.ipynb), use 10% as validation data.


	(X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data.npz', validation_split, verbose=True)

	c = axes_dict(axes)['C']
	n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


	# # CARE model
	# 
	# Before we construct the actual CARE model, we have to define its configuration via a `Config` object, which includes 
	# * parameters of the underlying neural network,
	# * the learning rate,
	# * the number of parameter updates per epoch,
	# * the loss function, and
	# * whether the model is probabilistic or not.
	# 
	# The defaults should be sensible in many cases, so a change should only be necessary if the training process fails.  
	# 
	# ---
	# 
	# <span style="color:red;font-weight:bold;">Important</span>: Note that for this notebook we use a very small number of update steps per epoch for immediate feedback, whereas this number should be increased considerably (e.g. `train_steps_per_epoch=400`) to obtain a well-trained model.



	config = Config(axes, n_channel_in=1, n_channel_out=1, train_steps_per_epoch=train_steps_per_epochs,train_epochs=train_epochs)

	# We now create a CARE model with the chosen configuration:
	model = CARE(config, model_name, basedir='models')



	# # Training
	# 
	# Training the model will likely take some time. We recommend to monitor the progress with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) (example below), which allows you to inspect the losses during training.
	# Furthermore, you can look at the predictions for some of the validation images, which can be helpful to recognize problems early on.
	# 
	# You can start TensorBoard from the current working directory with `tensorboard --logdir=.`
	# Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.
	# 
	# ![](http://csbdeep.bioimagecomputing.com/img/tensorboard_denoising3D.png)

	history = model.train(X,Y, validation_data=(X_val,Y_val))
	plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])
	

	with open('loss.csv', 'w') as f:
    		for key in history.history.keys():
        		f.write("%s,%s\n"%(key,history.history[key]))
	# Plot final training history (available in TensorBoard during training):


	# # Evaluation
	# 
	# Example results for validation images.

	

	if (plot_evaluation):
		plt.figure(figsize=(12,7))
		_P = model.keras_model.predict(X_val[:5])
		if config.probabilistic:
		    _P = _P[...,:(_P.shape[-1]//2)]
		plot_some(X_val[:5,:,:,:,0],Y_val[:5,:,:,:,0],_P[:,:,:,:,0],pmax=99.5)
		plt.suptitle('5 example validation patches\n'      
			     'top row: input (source),  '          
			     'middle row: target (ground truth),  '
			     'bottom row: predicted from source')
		fig1 = plt.gcf()
		fig1.savefig('/stornext/Img/data/prkfs1/m/Microscopy/Nina_Tubau/img_5epochs.png',transparent=True)


	# # Export model to be used with CSBDeep **Fiji** plugins and **KNIME** workflows
	# 
	# See https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji for details.

	# In[13]:


	model.export_TF()

if __name__ == '__main__':
	main()

