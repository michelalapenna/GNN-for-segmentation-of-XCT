# Geometric Deep Learning for enhanced quantitative analysis of Microstructures in X-ray Computed Tomography Data

## Abstract

Quantitative microstructural analysis of XCT 3D images is key for quality assurance of materials and components. In this paper we
implement a Graph Convolutional Neural Network (GCNN) architecture to segment a complex Al–Si Metal Matrix composite XCT volume (3D image). We train the model on a synthetic dataset and we assess its performance on both synthetic and experimental, manually labelled, datasets. 

Our simple GCNN shows a comparable performance to more standard machine learning methods, but uses a greatly reduced number of parameters, features low training time, and needs little hardware resources. Our GCNN thus achieves a cost-effective reliable segmentation.

## The repository

The `Example_Data` folder contains a 8x8x8 sub-volume of features and labels extracted from one of the 512x512x512 synthetic volumes used for training. In the same folder we provide the cross-section image of an experimental volume and two more images showing the ground truth labels of a synthetic slice and the corresponding predictions by our model.

In the folder `scripts` we provide the .pys including all the necessary imports, custom functions and classes needed to execute the code in the notebooks. We furthermore provide a different notebook for the training of the model, the testing on the synthetic dataset and the testing on the experimental dataset respectively.

The complete synthetic and experimental datasets used in the paper are available from the corresponding author upon request. The trained models are available from [this link](https://liveunibo-my.sharepoint.com/:f:/g/personal/ferdinando_zanchett2_unibo_it/EmIJsOuc311MqaonANsFLU4BUuXTnRbOWu0_5Yv33KptWg?e=rJZTgm).

## Installation of the correct environment

We report the commands used in conda to consistently install pytorch geometric.

```
conda install pytorch==1.11.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 

conda install pyg pytorch-cuda=11.7 -c pyg -c nvidia 
```
