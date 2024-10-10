# Graph Neural Networks for enhanced quantitative analysis of Microstructures in X-ray Computed Tomography Data

## Abstract

Quantitative microstructural analysis of XCT 3D images is key for quality assurance of materials and components. In this work we implement two different Graph Convolutional Neural Network (GCNN) architectures to segment a complex Al–Si Metal Matrix composite XCT volume (3D image). We train the model on a low-resembling synthetic dataset and we assess its performance on both synthetic and experimental, manually-labelled, data. To enhance the segmentation of experimental data, we fine-tune the trained model on an experimental sub-volume previously segmented by the trained model itself and then further manually-labelled. Fine-tuning largely increases the quality of the segmentation.

Our GCNN architectures show a comparable performance to state-of-the-art 3D Deep Convolutional Neural Network (3DCNN) and autoencoders, but use a greatly reduced number of parameters. Once the libraries employed for GCNNs will reach the same optimization level as the ones implementing usual 3DCNNs, this reduced number of trainable parameters will allow to cut the computational costs both in training and testing.

## Description of the two architectures

First, we trained a simple GCNN architecture on the non-augmented synthetic dataset, just as a proof of concept for the usage of GCNNs for this task. The encoding is obtained by a sequence of three GCLs, each followed by a non-linearity: GAT (Graph Attention Networks) [[1]](#1), GraphSage, and GCN (Graph Convolutional Network). The decoder consists of two linear layers separated by a non-linear activation function. The first GCL is a GAT layer, so that the attention coefficients are computed directly on the input features.

Even if this model correctly understands the semantic of microstructures, the performance gets considerably worse when it is trained on the augmented dataset. This is due to the fact that the model is too small and it is not able to learn equally well when the range of gray levels is enlarged. This lead us to build a second, larger, architecture, to train on the synthetic augmented dataset and further improve the segmentation on experimental data.

To build the second enlarged GCNN architecture, we adapt Vision Graph Neural Network (ViG) blocks. A ViG block consists of two main parts: the Grapher and the FFN (Feed Forward Network) modules. The Grapher module contains the Graph Convolutional layers, while the FFN module is a simple multi-layer perceptron with two fully-connected layers and it is introduced to further encourage the feature transformation and alleviate the typical phenomenon of over-smoothing in GCNNs. Indeed, introducing a FFN module after the Graph Convolutions interrupts the Message Passing flow and projects the nodes' embeddings into a larger space, thus avoiding over-smoothing when enlarging the architecture. The final architecture is built up by concatenating different ViG blocks.

Our enlarged GCNN architecture comprises two adapted ViG blocks. The Grapher module contains one initial 3DCNN layer, useful to extract richer and more significative features for the nodes, and a succession of three GCLs as in the simple GNN architecture, each followed by a non-linearity: one first GIN (Graph Isomorphism Network) layer with a 2-layer MLP, and two consecutive GraphSage layers. Batch normalization is applied after the 3DCNN and after each GCL layer in the Grapher module. On the other side, the FFN module is composed of two linear layers separated by a non-linear activation function. The decoder of the model is simply constituted by the final linear layer. In between the different modules, skip connections are inserted to help the convergence of the model. We stress that the size and the topology of the graph is not changing throughout the architecture. On the other hand, the dimension of the embedding is increasing until the final dense layer projects it on the number of classes.

![ViG_archi](https://github.com/user-attachments/assets/eda4ac76-b7bc-4914-b897-07f02ef421b5)

## References
<a id="1">[1]</a> 
Veli{\v{c}}kovi{\'{c}} Petar, Cucurull Guillem, Casanova Arantxa, Romero Adriana, Li{\`{o}} Pietro and Bengio Yoshua. 
Graph Attention Networks. 
International Conference on Learning Representations, 2018.

## The repository

In the folder `scripts` we include the .pys containing all the necessary imports and utils functions, together with the definition of the models and the Train/Test functions. We furthermore provide separate notebooks for training and testing.

The `Segmentation Examples` folder contains images showing the ground truth labels and the segmentation by our two models on experimental slices.

The `Samples Data` folder contains, instead, samples from the synthetic and experimental dataset. We provide sub-volumes of dimensions 150x150x150 voxels, with corresponding labels. In the case of experimental data, we do not provide a labelled sub-volume but only a manually-labelled slice from the sub-volume.

The complete synthetic and experimental datasets used in the paper are available from the corresponding author upon request.

## Installation of the correct environment

We report the commands used in conda to consistently install pytorch geometric.

```
conda install pytorch==1.11.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 

conda install pyg pytorch-cuda=11.7 -c pyg -c nvidia
```
