# Denoising 3D microscopy images with CSBDeep

This is an application of CSBDeep network for denoising microscopy images.

<img src="imgs/noisy.jpg" width="100px"/>
<img src="imgs/denoised.jpg" width="100px"/>

[Martin Weigert](https://www.biorxiv.org/content/10.1101/236463v1)

## Setup

### Prerequisites
- Linux or OSX

### Getting started
- Install python https://realpython.com/installing-python/
- Install tensorflow https://www.tensorflow.org/install/
```bash
# Current stable release for CPU-only
pip install tensorflow 
```
- Install CSBDeep package http://csbdeep.bioimagecomputing.com/doc/install.html
```bash
pip install csbdeep 
```
OR
Since installing TensorFlow with its dependencies (CUDA, cuDNN) can be challenging, there is a ready-to-use [Docker container](https://hub.docker.com/r/tboo/csbdeep_gpu_docker/) as an alternative to get started more quickly. 

- Clone this repository
```bash
git clone https://github.com/ninatubau/denoising.git
```
## Data

The dataset has to follow a particular structure as following:

<img src="imgs/dataset_structure.png" width="100px"/>
