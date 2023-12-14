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

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU, LeakyReLU, Unfold, Fold
from torch import Tensor

from tifffile import tifffile

import torch_geometric

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

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
