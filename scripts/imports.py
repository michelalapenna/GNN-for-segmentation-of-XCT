import time
import copy
import math
import csv
import networkx as nx
import numpy as np
from numpy import roots
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations
import random
from skimage.util.shape import view_as_windows, view_as_blocks
from skimage.segmentation import mark_boundaries, find_boundaries, slic
import os.path as osp

from tqdm import tqdm

import torch as torch
from torch import scatter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU, LeakyReLU, Unfold, Fold
from torch import Tensor
import scipy.spatial

import torch_cluster as torch_cluster

from tifffile import tifffile

import torch_geometric

#from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.inits import reset
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader, Data, Dataset
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, DynamicEdgeConv, EdgeConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch_geometric.nn import GraphUNet
#from torch_geometric.utils import dropout_edge

import torch_scatter
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from tqdm.notebook import tqdm

import torchmetrics
from torchmetrics import Accuracy, F1Score, Dice
from torchmetrics.functional import dice
from torchmetrics.classification import MulticlassF1Score

from collections import Counter
from sklearn.neighbors import NearestNeighbors
import gc
import GPUtil
