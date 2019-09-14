import csv
import pickle
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sample.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
PKL_PATH = './train.pkl'
MODE_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'
CLASSES = ('acqic','bacno','cano','conam','contp','csmcu','ecfg','etymd','flbmk',	
           'flg_3dsmk','fraud_ind','hcefg','insfg','iterm','locdt','loctm','mcc',
           'mchno','ovrlt','scity','stocn','stscd','txkey')
FEATURE = ('acqic','bacno','cano','conam','contp','csmcu','ecfg','etymd','flbmk',
           'flg_3dsmk','hcefg','insfg','iterm','locdt','loctm','mcc',
           'mchno','ovrlt','scity','stocn','stscd','txkey')

TARGET = ('fraud_ind')
##################

### Golbal Parameters ###
TestCol = 100
LEARNING_RATE = 0.001
######################### 
class Data():
    def __init__(self):
        self.X, self.y = [], []
        self.data, self.mean, self.std = {}, {}, {}
    def Read(self,path,nanStrategy='dropna'): 
        # strategy: mean, median, most_frequent, constant, dropna
        with open(path, newline='', encoding='utf-8') as csvfile:
            rows = np.array(list(csv.reader(csvfile)))[1:TestCol] ## 
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
        for i, key in enumerate(CLASSES):
            self.data[key] = rows[:,i].reshape(-1,1)

    def Encoding(self,enc,*args): 
        if enc == 'OneHot': # for discreting data
            for arg in args:
                self.data[arg] = pd.get_dummies(self.data[arg].reshape(-1,)).values
        elif enc == 'Label': # for sorting data
            for arg in args:
                lb = LabelEncoder()
                self.data[arg] = lb.fit_transform(self.data[arg]) 

#    def ReduceDim(self,method):
#        # method: PCA, MF, AutoEncoder

    def Normalize(self,*args,method='Rescaling',mean_or_max='None',\
                  std_or_min='None',trans=False):
        for arg in args:
            if trans is False:
                if method == 'Rescaling':
                    mean_or_max = np.max(self.data[arg],axis=0).reshape(1,-1)
                    std_or_min = np.min(self.data[arg],axis=0).reshape(1,-1) 
                elif method == 'Standardization':
                    mean_or_max = np.mean(self.data[arg],axis=0).reshape(1,-1)
                    std_or_min = np.std(self.data[arg],axis=0).reshape(1,-1)        
                self.mean[arg], self.std[arg] = mean_or_max, std_or_min
            elif trans is True:
                if method == 'Rescaling':
                    self.data[arg] = (self.data[arg] - std_or_min[arg]) / (mean_or_max[arg] - std_or_min[arg])
                elif method == 'Standardization':
                    self.data[arg] = (self.data[arg] - mean_or_max[arg]) / std_or_min[arg]
        if trans is False:
            return self.mean, self.std 
                    
    def Get_Feature(self,model,*args):
        for arg in args:
            self.X += [self.data[arg]]
        self.X = np.hstack(self.X)
        if model == 'cpu':
            self.X, self.y = self.X, self.data[TARGET]
        elif model == 'cuda':
            self.X = torch.FloatTensor(self.X)
            self.y = torch.LongTensor(self.data[TARGET])
        return self.X, self.y
    
    def Save_Pickle(self, name):
        file = open(name, 'wb')
        pickle.dump(self.data, file)
        file.close()
    def Load_Picke(self):
        return 
             
def Split_train_val(X,y,train_rate=0.7):
    if len(X) != len(y):
        return print("no match size!")
    perm = torch.randperm(len(y)) # Shuffle
    X, y = X[perm], y[perm]
    split_pos = int(np.round(len(y)*train_rate))
    return X[:split_pos], X[split_pos:], y[:split_pos], y[split_pos:]



raw = Data()
raw.Read(RAWDATA_PATH)
raw.Encoding('OneHot','acqic','csmcu')
mean, std = raw.Normalize('acqic','csmcu','bacno')
raw.Save_Pickle(PKL_PATH)
raw_X, raw_y = raw.Get_Feature('cuda',*FEATURE)

train_X, test_X, train_y, test_y = Split_train_val(raw_X, raw_y)

