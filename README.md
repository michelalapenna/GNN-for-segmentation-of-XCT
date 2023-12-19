# Geometric Deep Learning for enhanced quantitative analysis of Microstructures in X-ray Computed Tomography Data

Quantitative microstructural analysis of XCT 3D images is
key for quality assurance of materials and components. In this paper we
implement a Graph Convolutional Neural Network (GCNN) architecture
to segment a complex Al–Si Metal Matrix composite XCT volume (3D
image). We train the model on a synthetic dataset and we assess its
performance on both synthetic and experimental, manually labelled,
datasets. Our simple GCNN shows a comparable performance to more
standard machine learning methods, but uses a greatly reduced number
of parameters, features low training time, and needs little hardware
resources. Our GCNN thus achieves a cost-effective reliable segmentation.

The `Example_Data` folder contains a 8x8x8 sub-volume of features and labels extracted from one of the 512x512x512 synthetic volumes used for training. In the same folder we provide the image of a cross-section of an experimental volume and two more images showing the ground truth labels of a synthetic slice and the corresponding predictions by our model.


