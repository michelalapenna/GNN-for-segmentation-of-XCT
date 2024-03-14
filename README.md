# Geometric Deep Learning for enhanced quantitative analysis of Microstructures in X-ray Computed Tomography Data

## Abstract

Quantitative microstructural analysis of XCT 3D images is key for quality assurance of materials and components. In this work we
implement two different Graph Convolutional Neural Network (GCNN) architectures to segment a complex Al–Si Metal Matrix composite XCT volume (3D image). We train the model on a synthetic dataset and we assess its performance on both synthetic and experimental, manually labelled, datasets. 

Our GCNN architectures show a comparable performance to state-of-the-art 3D Deep Convolutional Neural Network (DCNN) and autoencoders, but use a greatly reduced number of parameters. Once the libraries employed for GCNNs will reach the same optimization level as the ones implementing usual DCNNs, this reduced number of trainable parameters will allow to cut the computational costs both in training and testing.


## The repository

The `Segmentation Examples` folder contains images showing the ground truth labels and the segmentation by our simple GCNN and ViG models (both for synthetic and experimental slices).

The `Samples Data` folder contains, instead, samples from the synthetic and experimental dataset. We provide subvolumes of dimensions 150x150x150 voxels, with corresponding labels. In the case of experimental data, we do not provide a labelled subvolume but only a manually-labelled slice from the subvolume.

In the folder `scripts` we include the .pys containing all the necessary imports, custom functions and classes needed to execute the code in the notebooks. We furthermore provide separate notebooks for training, testing on the synthetic dataset and testing on the experimental dataset respectively.

The complete synthetic and experimental datasets used in the paper are available from the corresponding author upon request. The trained models are available from [this link](https://liveunibo-my.sharepoint.com/:f:/g/personal/ferdinando_zanchett2_unibo_it/EmIJsOuc311MqaonANsFLU4BUuXTnRbOWu0_5Yv33KptWg?e=rJZTgm).

## Installation of the correct environment

We report the commands used in conda to consistently install pytorch geometric.

```
conda install pytorch==1.11.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 

conda install pyg pytorch-cuda=11.7 -c pyg -c nvidia 
```
