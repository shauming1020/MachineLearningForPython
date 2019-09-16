import csv
import pickle
import numpy as np
import pandas as pd
import torch 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

"""
    Data Management
"""

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


class Data():
    def __init__(self, data={}, target=[]):
        self.data = data
        self.target = target
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
        if 'train' in path:
            for i, key in enumerate(CLASSES):
                self.data[key] = rows[:,i].reshape(-1,1)
                self.target = rows[:,10].reshape(-1,1)
        elif 'test' in path:
            for i, key in enumerate(FEATURE):
                self.data[key] = rows[:,i].reshape(-1,1)        

    def Encoding(self,enc,*args): 
        if enc == 'OneHot': # for discreting data
            for arg in args:
                self.data[arg] = pd.get_dummies(self.data[arg].reshape(-1,)).values
        elif enc == 'Label': # for sorting data
            for arg in args:
                lb = LabelEncoder()
                self.data[arg] = lb.fit_transform(self.data[arg]) 
                
    def Split(self,md='split',rate=0.7):
        sp, sp_y = {}, []
        size = len(self.target)
        split_pos = int(np.round(size*rate))
        perm = np.random.permutation(size)  
        if md == 'split':
            for arg in self.data.keys():
                sp[arg] = self.data[arg][perm]
                sp_y = self.target[perm]
            for arg in self.data.keys():
                sp[arg] = sp[arg][split_pos:]
                sp_y = sp_y[split_pos:]
                self.data[arg] = self.data[arg][:split_pos] 
                self.target = self.target[:split_pos]
#        if md == 'sample':
        return Data(sp, sp_y)
               
    def Make_Torch(self,*args,return_y=True):   
        self.X, self.y = [], []
        for arg in args:
            self.X += [self.data[arg]]
        self.X = torch.FloatTensor(np.hstack(self.X))
        if return_y:
            self.y = torch.LongTensor(self.target)
    
    def Pickle(self, name, md='save'):
        if md == 'save':
            file = open(name, 'wb')
            pickle.dump(self.data, file)
            file.close()

#def ReduceDim(self,method):
#    # method: PCA, MF, AutoEncoder 

def Pad(data,mean,*args):
    for arg in args:
        if np.shape(data[arg])[1] < np.shape(mean[arg])[1]:
            pad_size = np.shape(mean[arg])[1] - np.shape(data[arg])[1]
            data[arg] = np.pad(data[arg], ((0,0),(0,pad_size)), 'constant')

def Normalize(data,*args,method='Rescaling',mean_or_max='None',\
              std_or_min='None',trans=False):  
    E = 1e-09
    if trans is False:
        mean_or_max, std_or_min = {}, {}
    for arg in args:
        if trans is False:
            if method == 'Rescaling':
                mean_or_max[arg] = np.max(data[arg],axis=0).reshape(1,-1)
                std_or_min[arg] = np.min(data[arg],axis=0).reshape(1,-1) 
            elif method == 'Standardization':
                mean_or_max[arg] = np.mean(data[arg],axis=0).reshape(1,-1)
                std_or_min[arg] = np.std(data[arg],axis=0).reshape(1,-1)               
        if method == 'Rescaling':
            data[arg] = (data[arg] - std_or_min[arg]) / (mean_or_max[arg] - std_or_min[arg] +E)
        elif method == 'Standardization':
            data[arg] = (data[arg] - mean_or_max[arg]) / (std_or_min[arg] +E)
    if trans is False:
        return mean_or_max, std_or_min
            
def Split_train_val(X,y,split_rate=0.7):
    if len(X) != len(y):
        return print("no match size!")
    perm = torch.randperm(len(y)) # Shuffle
    X, y = X[perm], y[perm]
    split_pos = int(np.round(len(y)*split_rate))
    return X[:split_pos], X[split_pos:], y[:split_pos], y[split_pos:]


