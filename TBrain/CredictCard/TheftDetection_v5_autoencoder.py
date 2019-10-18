import csv
import time
import pickle
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import MyAutoEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
args_feature = args_discrete + args_continously
###############################################################################

############################ Golbal Parameters ################################
E = 1e-12
WORKERS = 0
TestCol = 10240
ENSEMBLE_NUM = 1
LEARNING_RATE = 0.002
EPOCHS = 64
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
            rows = np.array(list(csv.reader(csvfile)))[1:TestCol] ## 
#            rows = np.array(list(csv.reader(csvfile)))[1:]
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs """
    lr = LEARNING_RATE * (LR_DECAY ** (epoch // PER_EPOCHS_TO_DECAY_LR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Fit(iteration,train_set,val_set,model,loss,optimizer,batch_size,epochs):
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=WORKERS)   
    
    loss_history = []
    lmbda = 0.1
    budget = 0.3
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        adjust_learning_rate(optimizer, epoch)
        
        train_loss = 0.
        confidence_loss = 0. 
        val_loss = 0.
        
        model.train() # Switch to train mode
        for i, data in enumerate(train_loader):
            # compute output
            train_pred, confidence = model(data[0].cuda())   
            # Make sure we don't have any numerical instability
            eps = 1e-12
            train_pred = torch.clamp(train_pred, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
            
            train_loss = loss(train_pred, data[0].cuda())
            confidence_loss = torch.mean(-torch.log(confidence))
            total_loss = train_loss + (lmbda * confidence_loss)
            
            if budget > confidence_loss.data:
                lmbda = lmbda / 1.01
            elif budget <= confidence_loss.data:
                lmbda = lmbda / 0.99

            #compute gradient and do step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += train_loss.item()
            confidence_loss += confidence_loss.item()
     
#        model.eval() # Switch to evaluate mode
#        for i, data in enumerate(val_loader):
#            # compute output
#            with torch.no_grad():
#                val_pred = model(data[0].cuda())
#                batch_loss = loss(val_pred, data[0].cuda())
#    
#            val_pred = np.argmax(val_pred.float().cpu().data.numpy(), axis=1)
#            batch_loss = batch_loss.float()
#    
#            val_loss += batch_loss.item()
            
        print('[%03d/%03d] %2.2f sec(s) Train Loss: %3.3f | Val loss: %3.3f' % \
                (epoch + 1, epochs, time.time()-epoch_start_time, \
                 train_loss, val_loss))
        
        loss_history.append((train_loss,val_loss))

    return np.asarray(loss_history)

def load_all_models(n_numbers, in_dim):
    all_models = list()
    for i in range(n_numbers):
        filename = MODEL_PATH+'/'+str(i+1)+'_model.pth'
        model = MyAutoEncoder.My_AutoEncoder(in_dim).cuda()
        model.load_state_dict(torch.load(filename))
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def _Encoding():
    ## Read Normaly Raw Data
    normaly = Data() 
    normaly.Read(RAWDATA_PATH,nanStrategy='dropna',take='normaly')   
    
    ## Encoding and Dim Reduction
    print('Encoding with discrete data...')
    enc_members = {}
    for arg in args_discrete:
        enc = OneHotEncoder(handle_unknown='ignore')
        normaly.data[arg] = enc.fit_transform(normaly.data[arg]).toarray() + E # sparse matrix to array
        enc_members[arg] = enc

#    ## Embedding 
#    for arg in args_discrete:
#        em_in = np.shape(normaly.data[arg])[1]
#        embedding = nn.Embedding(em_in, int(em_in/2))
#        normaly.data[arg] = embedding(torch.LongTensor(normaly.data[arg]))
#        normaly.data[arg] = normaly.data[arg].view(normaly.data[arg].size(0),-1) # flatten
#        normaly.data[arg] = normaly.data[arg].data.numpy()
    
    ## Normalize - Standardization
    scaler_members = {}
    for arg in args_feature:
        scaler = StandardScaler(copy=False).fit(normaly.data[arg])
        normaly.data[arg] = scaler.transform(normaly.data[arg])
        scaler_members[arg] = scaler

    
    ## Unspervised learning - PCA(dimension reduction) with batch
    print('PCA to dimension reduction...')
    pca_members = {}
    from sklearn.decomposition import IncrementalPCA
    for arg in args_discrete:
        print(arg,'before pca', np.shape(normaly.data[arg]))
        N_COMP = int(np.shape(normaly.data[arg])[1] / 2) # 
        ipca = IncrementalPCA(n_components=N_COMP, copy=False, batch_size=BATCH_SIZE)
        normaly.data[arg] = ipca.fit_transform(normaly.data[arg])
        print(arg,'after dim reduce', np.shape(normaly.data[arg]))
        pca_members[arg] = ipca

    ## Make the Torch
    X_discrete, X_continously = [], []
    for arg in args_discrete:
        X_discrete += [normaly.data[arg]]
    X_discrete = torch.FloatTensor(np.hstack(X_discrete))
  
    for arg in args_continously:
        X_continously += [normaly.data[arg]]
    X_continously = torch.FloatTensor(np.hstack(X_continously))  
    
    
    X_train = X_train[torch.randperm(X_train.__len__())] # Shuffle
    split_pos = int(np.round(X_train.__len__()*0.8))
    X_train, X_val = X_train[:split_pos], X_train[split_pos:]
        
    train_set, val_set = TensorDataset(X_train), TensorDataset(X_val)
   
    print("Building Model...")
    model = MyAutoEncoder.My_AutoEncoder(in_dim=X_train.size()[1]).cuda()
    loss = nn.MSELoss() # The criterion combines nn.LogSoftmax()
    loss = loss.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)   
    
    ## Training infomation
    print('Total parameters:',sum(p.numel() for p in model.parameters()),\
          ', Trainable parameters:comp',sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Training set size:',len(X_train),', Validation set size:',len(X_val))
    
    print("Fitting Model...")
    loss_history = Fit(1,train_set,val_set,model,loss,optimizer,batch_size=BATCH_SIZE,epochs=EPOCHS)
    
    return model, loss_history
   
    















#    ## 
#    raw = Data() 
#    raw.Read(RAWDATA_PATH,nanStrategy='dropna',take='anormaly')   
#    
#    ## Encoding and Dim Reduction
#    print('Encoding with discrete data...')
#    for arg in args_discrete:
#        enc = enc_members[arg]
#        raw.data[arg] = enc.transform(raw.data[arg]).toarray() + E # sparse matrix to array
#
#    ## Normalize - Standardization
#    for arg in args_feature:
#        scaler = scaler_members[arg]
#        raw.data[arg] = scaler.transform(raw.data[arg])
#  
#    ## Unspervised learning - PCA(dimension reduction) with batch
#    print('PCA to dimension reduction...')
#    for arg in args_discrete:
#        print(arg,'before pca', np.shape(raw.data[arg]))
#        ipca = pca_members[arg]
#        raw.data[arg] = ipca.transform(raw.data[arg])
#        print(arg,'after dim reduce', np.shape(raw.data[arg]))
# 
#    ## Make the Torch
#    X_anormaly = []
#    for arg in args_feature:
#        X_anormaly += [raw.data[arg]]        
#    X_anormaly = torch.FloatTensor(np.hstack(X_anormaly))
#    
#    anormaly_set = TensorDataset(X_anormaly)
#    anormaly_loader = DataLoader(anormaly_set, batch_size=1, num_workers=WORKERS)
#    
#    pred, anormaly_loss = [], []
#    for data in anormaly_loader:
#        anormaly_pred = model(data[0].cuda())
#        anormaly_loss += [float(loss(anormaly_pred, data[0].cuda()).cpu())]
#        pred += [anormaly_pred]

