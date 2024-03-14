# Geometric Deep Learning for enhanced quantitative analysis of Microstructures in X-ray Computed Tomography Data

## Abstract

Quantitative microstructural analysis of XCT 3D images is key for quality assurance of materials and components. In this work we
implement two different Graph Convolutional Neural Network (GCNN) architectures to segment a complex Al–Si Metal Matrix composite XCT volume (3D image). We train the model on a synthetic dataset and we assess its performance on both synthetic and experimental, manually labelled, datasets. 

Our GCNN architectures show a comparable performance to state-of-the-art 3D Deep Convolutional Neural Network (DCNN) and autoencoders, but use a greatly reduced number of parameters. Once the libraries employed for GCNNs will reach the same optimization level as the ones implementing usual DCNNs, this reduced number of trainable parameters will allow to cut the computational costs both in training and testing.

## Description of the two architectures

First, we trained a simple GCNN architecture on the non-augmented synthetic dataset, just as a proof of concept for the usage of GCNNs for this task. The encoding is obtained by a sequence of three GCLs, each followed by a non-linearity: GAT (Graph Attention Networks), GraphSage, and GCN (Graph Convolutional Network). The decoder consists of two linear layers separated by a non-linear activation function. The first GCL is a GAT layer, so the attention coefficients are computed directly on the input features.

Even if this model understands the semantic of microstructures correctly, the performance gets considerably worse when it is trained on the augmented dataset. This is due to the fact that the model is too small and it is not able to learn equally well when the range of gray levels is enlarged. This lead us to build a second,larger, architecture, to train on the synthetic augmented dataset and further improve the segmentation on experimental data.

To build the second enlarged GCNN architecture, we adapt Vision Graph Neural Network (ViG) blocks. A ViG block consists of two main parts: the Grapher and the FFN (Feed Forward Network) modules. The Grapher module contains the Graph Convolutional layers, while the FFN module is a simple multi-layer perceptron with two fully-connected layers and it is introduced to further encourage the feature transformation and alleviate the typical phenomenon of over-smoothing in GCNNs. Indeed, introducing a FFN module after the Graph Convolutions interrupts the Message Passing flow and projects the nodes' embeddings into a larger space, thus avoiding over-smoothing when enlarging the architecture. The final architecture is built up by concatenating different ViG blocks.

Our enlarged GCNN architecture comprises two adapted ViG blocks. The Grapher module contains one initial 3DCNN layer, useful to extract richer and more significative features for the nodes, and a succession of three GCLs as in the simple GNN architecture, each followed by a non-linearity: GAT (Graph Attention Networks), GraphSage, and GCN (Graph Convolutional Network). Likewise, the FFN module is composed of two linear layers separated by a non-linear activation function. The outputs of the two adapted ViG blocks are then summed together before passing through a final linear layer. This residual connection has the aim of helping the convergence of the model.

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
