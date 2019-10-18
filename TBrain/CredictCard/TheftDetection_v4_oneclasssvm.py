import csv
import time
import pickle
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

################################## Parameters #################################
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/submission.csv'
X_train_PKL = './dataset/X_train.pkl'
Y_train_PKL = './dataset/Y_train.pkl'
X_ob_PKL = './dataset/X_ob.pkl'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODEL_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'
CLASSES = ('acqic','bacno','cano','conam','contp','csmcu','ecfg','etymd','flbmk',	
           'flg_3dsmk','fraud_ind','hcefg','insfg','iterm','locdt','loctm','mcc',
           'mchno','ovrlt','scity','stocn','stscd','txkey')
FEATURE = ('acqic','bacno','cano','conam','contp','csmcu','ecfg','etymd','flbmk',
           'flg_3dsmk','hcefg','insfg','iterm','locdt','loctm','mcc',
           'mchno','ovrlt','scity','stocn','stscd','txkey')
TARGET = ('fraud_ind')
###############################################################################

############################# FEATURE SELECTING ###############################
# mchno, acqic will make the memory error
args_discrete = 'contp','etymd','mcc','ecfg','insfg','stocn','stscd',\
            'ovrlt','flbmk','hcefg','csmcu','flg_3dsmk'
args_continously = 'locdt','loctm','conam','iterm'
args_feautre = args_discrete + args_continously
###############################################################################

############################ Golbal Parameters ################################
E = 1e-10
WORKERS = 0
TestCol = 1000
ENSEMBLE_NUM = 1
LEARNING_RATE = 0.002
EPOCHS = 16
BATCH_SIZE = 256
PER_EPOCHS_TO_DECAY_LR = 32
LR_DECAY = 0.8
REG = 0 # L2-Norm
WEIGHT_DECAY = 1e-09
PATIENCE = 64
###############################################################################

class Data():
    def __init__(self, data={}, target=[]):
        self.data = data
        self.target = target
    def Read(self,path,nanStrategy='dropna',take=None): 
        # strategy: mean, median, most_frequent, constant, dropna
        with open(path, newline='', encoding='utf-8') as csvfile:
#            rows = np.array(list(csv.reader(csvfile)))[1:TestCol] ## 
            rows = np.array(list(csv.reader(csvfile)))[1:]
            where_N, where_Y, where_nan = rows=='N', rows=='Y', rows==''
            rows[where_N], rows[where_Y], rows[where_nan] = 0, 1, np.nan
            rows = rows.astype(float)
            if nanStrategy == 'dropna':
                rows = rows[~np.isnan(rows).any(axis=1)]
            elif nanStrategy is None:
                rows = rows
            else:
                imp = SimpleImputer(strategy=nanStrategy)
                rows = imp.fit_transform(rows)
        if 'train' in path:
            if take == 'normaly':
                rows = rows[rows[:,10] == 0]
            elif take == 'anormaly':
                rows = rows[rows[:,10] == 1]
            for i, key in enumerate(CLASSES):
                if key is TARGET:
                    self.target = rows[:,10].reshape(-1,1)
                else:
                    self.data[key] = rows[:,i].reshape(-1,1)
        elif 'test' in path:
            for i, key in enumerate(FEATURE):
                self.data[key] = rows[:,i].reshape(-1,1)

def Split_train_val(x,y,train_rate=0.8):
    if len(x) != len(y):
        return print("no match size!")
    perm = torch.randperm(len(y)) # Shuffle
    x, y = x[perm], y[perm]
    split_pos = int(np.round(len(y)*train_rate))
    return x[:split_pos], x[split_pos:], y[:split_pos], y[split_pos:]

def Save_PKL(data, path):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()
    
def Load_PKL(path):
    with open(path, 'rb') as file:
        return pickle.load(file)




def _Ensemble():
    ## Read Raw Data
    raw_normaly = Data() 
    raw_normaly.Read(RAWDATA_PATH,nanStrategy='dropna',take='normaly') 
    X_train = []
    for arg in args_continously:
        X_train += [raw_normaly.data[arg]]
    
    raw_anormaly = Data()
    raw_anormaly.Read(RAWDATA_PATH,nanStrategy='dropna',take='anormaly') 
    X_outliers = []
    for arg in args_continously:
        X_outliers += [raw_anormaly.data[arg]]

    X_train = torch.FloatTensor(np.hstack(X_train))
    X_outliers = torch.FloatTensor(np.hstack(X_outliers))    
    
    from sklearn import svm
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01)    
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
if __name__ == '__main__':    
    _Ensemble()