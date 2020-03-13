# Denoising 3D microscopy images with CSBDeep

This is an application of CSBDeep network for **denoising microscopy images**. It is also possible to load other pre-trained models (.h5 files) to denoise images. this application can be run through the [terminal](#Command) or through the [interface](#Interface). Finally, it also gives a method to evaluate denoising through different metrics such as signal-to-noise ratio, structural similarity index and jaccard index.


<img src="imgs/clean.gif" width="410px" align="left" alt="After denoising" title="Before denoising"/>
<img src="imgs/noisy.gif" width="410px" align="right" alt="After denoising" title="After denoising"/>   
    
</br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> </br> 


## Setup

### Prerequisites
- Linux or OSX

### Getting started
- Install [python](https://realpython.com/installing-python/)

- Clone this repository
```bash
git clone https://github.com/ninatubau/denoising.git
```

- Install package (numpy, napari, tensorflow, [csbdeep](http://csbdeep.bioimagecomputing.com/doc/install.html) etc.) 
```bash
pip install -r requirements.txt 
```
OR
since installing TensorFlow with its dependencies (CUDA, cuDNN) can be challenging, a Dockerfile is provided as an alternative to get started more quickly.
 
- Installing from Dockerfile
    - Install Docker and nvidia-docker 
    - Create a docker image from the dockerfile. From the working directory of this repo:
    
    ```
    nvidia-docker build -t denoising:1.0 .
    ``` 
    This will take care of all the required drivers and dependencies provided your current CUDA drivers are up to date. 
    
    Running via the docker container requires a few extra arguments to allow the GUI. <B>Note unfortunately at this point the napari based image viewer is not available when using docker.</B> 
    
    It is also often convenient to mount a data directory to the container to allow ease of data transfer - you can put it within the denoising folder but we find this a more convenient solution: 
  ```
  nvidia-docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)":/denoising/ -v <YOURDATAPATH>:/data/ -e DISPLAY=unix:1 -u $(id -u):$(id -g) denoising:1.0 /bin/bash
  ```
    

## Data

The dataset has to follow a particular structure as following:

<img src="imgs/dataset_structure.png" width="900px" align="center"/>

## Interface

<img src="imgs/interface.gif" width="550px" align="right" alt="Interface" title="Interface"/>

```bash
cd interface
python interfaceCARE.py
```
</br> </br> </br> </br> </br> </br> </br> </br> 

## Terminal 

### Data preparation and training

Preparation of the data and training on noise and clean images

Arguments:
- data_path: Path to your input data: noisy (low intensity) and clean(high intensity) folder with .tif files
- *axes: Axes to indicate the semantic order of the images. Examples : ZYX, CXY ..., default XYZ* 
- *validation_split: Ratio of validation data for training, default 0.1*
- *train_steps_per_epochs: Number of training steps per epochs, default 100*
- *train_epochs: Number of epochs, default 10*
- *model_name: Name of the model saved in the models folder, default my_model*
- *patch_size: Size of patches to train the model, default (16,16,64)*

*optional arguments*

Outputs: 
- Plot of noisy/clean data
- Plot of loss
- Creation CSV file *loss.csv* with values of loss

```bash
python training.py path/to/dataset --optional_parameter value
```

### Prediction

Prediction of noisy images -> DENOISING

Arguments:
- path_data: Path to to_predict folder data
- name_model: Name of the model to use as trained model 
- *n_tiles: Number of tiles for evey image axes to avoid out of memory issues, default (1,4,4)*
- *axes: Axes to indicate the semantic order of the images. Examples : ZYX, CXY ..., default XYZ*
- *plot_prediction: Boolean value to plot some images after the prediction, default False*
- *filter_data: Filter dat the you want to predict. Examples: all, ch0, ch1, default all*


*optional arguments*

Output:
- Plot of evaluated images

```bash
python prediction.py path/to/data/to_predict --optional_parameter value
```


### Measure

Measurement of the results using Signal to noise ratio, structural similarity index and jaccard index

Arguments:
- data_path: Path to your data

Outputs:
- Box plot comparing noisy/ground_truth and result/ground_truth based on several metrics

```bash
python measure.py path/to/data/dataset 
```

## TO DO
- [x] Documentation
- [x] Link website with results (images and videos)
- [x] Clean code
- [ ] Finish interface


## Acknowledgment

Code borrows heavily from [CSBDeep](https://github.com/CSBDeep/CSBDeep) published on [Content-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy](https://www.biorxiv.org/content/10.1101/236463v1)
