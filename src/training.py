"""Neural network training for denoising lattice light sheet microscopy data.

This python code demonstrates training a CARE model for a 3D denoising task calling the datagen code that saves to disk to the file 'data/[data_name].npz'.
More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

This script requires 'os','sys','numpy','matplotlib','tifffile' and csbdeep modules to be imported.

    * parser_init - initialises the script parameters
    * load - loads the trainig data through data_generation
    * train - trains the neural network
    * save_results - save the results of the loss in a .csv file
"""

# from __future__ import print_function, unicode_literals, absolute_import, division
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from .datagen import data_generation
import argparse
import gc


parser = argparse.ArgumentParser(
    description='Data generation and training. The data folder contains 4 folders with tif files of the same name: noisy(input data), clean (ground truth), to_predict and predicted.')
parser.add_argument('data_path', type=str,
                    help='Path to your input data: noisy (low intensity) and clean(high intensity) folder with .tif files')
parser.add_argument('--axes', type=str, default='XYZ',
                    help='Axes to indicate the semantic order of the images axes. Examples : ZYX, CXY ... ')
parser.add_argument('--validation_split', type=int, default=0.1, help='Ratio of validation data for training')
parser.add_argument('--train_steps_per_epoch', type=int, default=1, help='Number of training steps per epochs')
parser.add_argument('--train_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--model_name', type=str, default='my_model', help='Name of the model to save')
parser.add_argument('--patch_size', nargs='+', type=int, default=(16, 64, 64),
                    help='Patch size for data generation, example (16,64,64)')
parser.add_argument('--probabilistic', type=bool, default=False, help='Parameter of the NN')
parser.add_argument('--train_batch_size', type=int, default=16, help='Parameter of the NN')
parser.add_argument('--train_learning_rate', type=float, default=0.0004, help='Parameter of the NN')
parser.add_argument('--unet_last_activation', type=str, default='linear', help='Parameter of the NN')
parser.add_argument('--unet_n_depth', type=int, default=2, help='Parameter of the NN')
parser.add_argument('--unet_n_first', type=int, default=32, help='Parameter of the NN')
parser.add_argument('--unet_residual', type=bool, default=True, help='Parameter of the NN')
parser.add_argument('--csv_file', type=str, default='loss.csv', help='CSV file saved after training with parameters')

parser.add_argument('--data_name', type=str, default='RBC_data',
                    help='Name of the .npz file generated with training data patches')


def parser_init(parser):
    """Initialises the arguments of the script with argparse

    Parameters
    ----------
    parser : argparse
        The input arguments to run the script

    Returns
    -------
    path_data
    axes
    validation_split
    train_steps_per_epoch
    train_epochs
    model_name
    patch_size
    """
    args = parser.parse_args()
    path_data = args.data_path
    axes = args.axes
    model_name = args.model_name
    validation_split = args.validation_split
    probabilistic = args.probabilistic,
    kwargs = {'unet_residual': args.unet_residual, 'unet_n_depth': args.unet_n_depth,'unet_n_first': args.unet_n_first, 'unet_last_activation': args.unet_last_activation,'train_steps_per_epoch': args.train_steps_per_epoch, 'train_epochs': args.train_epochs,  'train_batch_size': args.train_batch_size, 'train_learning_rate':args.train_learning_rate}
    patch_size = tuple(args.patch_size)
    data_name = args.data_name
    csv_file = args.csv_file

    return path_data, axes, validation_split, model_name, patch_size, data_name, csv_file, probabilistic, kwargs


def load(path_data, axes, validation_split, patch_size, data_name):
    """Loads the data patches to train.

    Parameters
    ----------
    path_data : str
        Path to input data
    axes : str
        Semantic order of the axis in the image
    validation_split : float
        Ratio of data kept for validation
    patch_size : tuple
        Size of the patches

    Returns
    -------
    X,Y : np.array
        Input data X for training, Ground truth Y for training
    X_val, Y_val : np.array
        Input data X_val for validation, Ground truth Y_val for validation
    """
    # limit GPU available memory
    # limit_gpu_memory(fraction=1/2)

    #data_generation(path_data, axes, patch_size, data_name)
    (X, Y), (X_val, Y_val), axes = load_training_data('data/' + data_name + '.npz', validation_split, verbose=True)
    return (X, Y), (X_val, Y_val)


def train(X, Y, X_val, Y_val, axes, model_name, csv_file,probabilistic, validation_split, patch_size, **kwargs):
    """Trains CARE model with patches previously created.

    CARE model parameters configurated via 'Config' object:
      * parameters of the underlying neural network,
      * the learning rate,
      * the number of parameter updates per epoch,
      * the loss function, and
      * whether the model is probabilistic or not.

    Parameters
    ----------
    X : np.array
        Input data X for training
    Y : np.array
        Ground truth data Y for training
    X_val : np.array
        Input data X for validation
    Y_val : np.array
        Ground truth Y for validation
    axes : str
        Semantic order of the axis in the image
    train_steps_per_epoch : int
        Number of training steps per epochs
    train_epochs : int
        Number of training epochs
    model_name : str
        Name of the model to be saved after training

    Returns
    -------
    history
        Object with the loss values saved
    """
    config = Config(axes, n_channel_in=1, n_channel_out=1, probabilistic=probabilistic,  allow_new_parameters=True, **kwargs)
    model = CARE(config, model_name, basedir='models')

    # # Training
    # [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) from the current working directory with `tensorboard --logdir=.`
    # Then connect to [http://localhost:6006/](http://localhost:6006/) with your browser.

    history = model.train(X, Y, validation_data=(X_val, Y_val))
    # df = pd.DataFrame()
    # df['probabilistic'] = [config.probabilistic]
    # df['batch_size'] = config.train_batch_size
    # df['train_epochs'] = config.train_epochs
    # df['lr'] = config.train_learning_rate
    # df['train_steps_per_epoch'] = config.train_steps_per_epoch
    # df['unet_last_activation'] = config.unet_last_activation
    # df['unet_n_depth'] = config.unet_n_depth
    # df['unet_n_first'] = config.unet_n_first
    # df['unet_residual'] = config.unet_residual
    # df['patch_size'] = str(patch_size)
    # df['validation_split'] = validation_split
    # last_value = len(history.history['loss']) - 1
    # #dict = kwargs
    # #data = dict.update({'loss':history.history['loss'][last_value], })
    # df['val_loss'] = history.history['val_loss'][last_value]
    # df['loss'] = history.history['loss'][last_value]
    # df['mse']=history.history['mse'][last_value]
    # df['mae']=history.history['mae'][last_value]
    # df['val_mse']=history.history['val_mse'][last_value]
    # df['val_mae']=history.history['val_mae'][last_value]
    # #df = pd.DataFrame.from_dict(data)
    # #print('File saved in:', os.getcwd()+'/'+csv_file)
    # df.to_csv('/data/'+csv_file)
    model.export_TF()
    return history

def save_results(history):
    """ Saves the loss values for a trained model in a csv file. """

    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])

    # save model results in loss.csv file
    with open('loss.csv', 'w') as f:
        for key in history.history.keys():
            f.write("%s,%s\n" % (key, history.history[key]))

    # Export model to be used with CSBDeep **Fiji** plugins and **KNIME** workflow
    gc.collect()


if __name__ == '__main__':
    path_data, axes, validation_split, model_name, patch_size, data_name, csv_file,probabilistic,  kwargs = parser_init(parser)
    (X, Y), (X_val, Y_val) = load(path_data, axes, validation_split, patch_size, data_name)
    history = train(X, Y, X_val, Y_val, axes, model_name, csv_file, probabilistic, validation_split, patch_size,**kwargs)
    #save_results(history)
