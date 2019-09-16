import csv
import pickle
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
        
def PCA():
    return 

def SVD():
    return

def Matrix_Factorization():
    return

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, code_dim):
        super(AutoEncoder, self).__init__()
        self.in_dim = in_dim
        self.code_dim = code_dim
        self.encoder = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, 16),
                nn.ReLU(True),
                nn.Linear(16, code_dim)
                )
        self.decoder = nn.Sequential(
                nn.Linear(code_dim, 16),
                nn.ReLU(True), 
                nn.Linear(16, 64),
                nn.ReLU(True),
                nn.Linear(64, 128),
                nn.ReLU(True),
                nn.Linear(128, in_dim), 
                nn.Tanh()
                )
    def forward(self, x):
        x = self.encoder(x)
        self.code = x
        x = self.decoder(self.code)
        return x