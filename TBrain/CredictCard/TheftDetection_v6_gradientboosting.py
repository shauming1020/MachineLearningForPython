import csv
import time
import pickle
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

################################## Parameters #################################
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/submission.csv'
X_train_PKL = './dataset/X_train_GDBT.pkl'
Y_train_PKL = './dataset/Y_train_GDBT.pkl'
X_ob_PKL = './dataset/X_ob_GDBT.pkl'
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
args_feature = args_discrete + args_continously
###############################################################################

############################ Golbal Parameters ################################
E = 1e-12
WORKERS = 0
TestCol = 102400
### GDBT ###
LOSS = 'deviance'
LEARNING_RATE = 0.1
N_ESTIMATORS = 128
MAX_DEPTH = 128
SUBSAMPLE = 2
CRITERION = 'friedman_mse'
MIN_SAMPLES_SPLIT = 8
MIN_SAMPLES_LEAF = 4
MAX_FEATURES = None

###############################################################################

class Data():
    def __init__(self, data={}, target=[]):
        self.data = data
        self.target = target
    def Read(self,path,nanStrategy='dropna'): 
        # strategy: mean, median, most_frequent, constant, dropna
        with open(path, newline='', encoding='utf-8') as csvfile:
#            rows = np.array(list(csv.reader(csvfile)))[1:TestCol] ## 
            rows = np.array(list(csv.reader(csvfile)))[1:]
            where_N, where_Y, where_nan = rows=='N', rows=='Y', rows==''
            rows[where_N], rows[where_Y], rows[where_nan] = -1, 1, np.nan
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
                if key is TARGET:
                    self.target = np.where(rows[:,10]==0,-1,rows[:,10]).reshape(-1,1)
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

def DecisionTree():
#    ## Read Normaly Raw Data
#    raw = Data() 
#    raw.Read(RAWDATA_PATH,nanStrategy='dropna')
#    
#    ## Make the Torch
#    X_train, y_train = [], []
#    for arg in args_feature:
#        X_train += [raw.data[arg]]
#    X_train, y_train = torch.FloatTensor(np.hstack(X_train)), torch.LongTensor(raw.target).view(-1,)
#    
#    ## Save pkl
#    Save_PKL(X_train,X_train_PKL)
#    Save_PKL(y_train,Y_train_PKL)
    ## Load pkl
    X_train = Load_PKL(X_train_PKL)
    y_train = Load_PKL(Y_train_PKL)    
    
    X_train, X_test, y_train, y_test = Split_train_val(X_train, y_train, train_rate=0.8)
    print('X_train size',X_train.size(),'y_train size',y_train.size())    
    print('Classifying...')
    tStart = time.time()
    
    print('GDBT')
    clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, 
                                     learning_rate=LEARNING_RATE,
                                     max_depth=MAX_DEPTH, 
                                     random_state=0,
                                     min_samples_split=MIN_SAMPLES_SPLIT,
                                     min_samples_leaf=MIN_SAMPLES_LEAF,
                                     max_features=MAX_FEATURES,
                                     verbose=1,
                                     ).fit(X_train, y_train)
#    print('RF')
#    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS).fit(X_train, y_train)
    
    tEnd = time.time()
    
    y_pred = clf.predict(X_test)
    print("It cost %3f sec" % (tEnd - tStart))
    print('Acc',clf.score(X_test, y_test))
    print('F1_score',f1_score(y_test, y_pred,average='macro', labels=np.unique(y_pred)))

    print("======================= Deal with T-Brain test data... =======================")
    ob = Data()
    ob.Read(OBSERVE_PATH,'mean')
     
    ## Make the Torch
    X_ob = []
    for arg in args_feature:
        X_ob += [ob.data[arg]]
        
    X_ob = torch.FloatTensor(np.hstack(X_ob))
    
    print('Submission...')
    submission = clf.predict(X_ob)
    with open(SUBMISSION_PATH, 'w') as submissionFile:
        submissionFile.write('txkey,fraud_ind\n') 
        for i,value in enumerate(submission):
            submissionFile.write('%d,%d\n' %(ob.data['txkey'][i], value))
    
    print('======================= Writing Complete! =======================')     

if __name__ == '__main__':    
    DecisionTree()