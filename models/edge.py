import copy
import pickle
import numpy as np
import pandas as pd
import torch
from torchinfo import summary

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import os

import pdb
import time

class Edge(object):
    # constructor
    def __init__(self, n_clients, bw_ec, ):
        # num of client
        self.n_clients = 20
        # edge to cloud bw
        self.bw_ec
        # edge computation speed (ignore since only aggregation)
        #self.comp
        # layer 1 glob model
        self.net_glob

    def train() -> :
        