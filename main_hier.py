#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import os

import pdb
import time
from edge import Edge

if __name__ == '__main__':
    # reproduce randomness
    torch.manual_seed(1001)
    np.random.seed(1001)
    n_edge = 5
    
    for edge in range(n_edge):
        
    
    