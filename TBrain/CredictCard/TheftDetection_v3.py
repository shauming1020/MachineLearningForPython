import csv
import time
import pickle
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
#import Classifier
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
    def Read(self,path,nanStrategy='dropna'): 
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
    
    loss_history, f1_score_history = [], []
    best_f1_score = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        adjust_learning_rate(optimizer, epoch)
        
        train_f1_score = 0.0
        train_loss = 0.0
        val_f1_score = 0.0
        val_loss = 0.0
    
        model.train() # Switch to train mode
        for i, data in enumerate(train_loader):
            # compute output
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            
            #compute gradient and do step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_pred = np.argmax(train_pred.float().cpu().data.numpy(),axis=1)
            loss = loss.float()
            
            train_f1_score += f1_score(data[1].numpy(), train_pred,\
                                       average='macro', labels=np.unique(train_pred))
            train_loss += batch_loss.item()
     
        model.eval() # Switch to evaluate mode
        for i, data in enumerate(val_loader):
            # compute output
            with torch.no_grad():
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())
    
            val_pred = np.argmax(val_pred.float().cpu().data.numpy(), axis=1)
            batch_loss = batch_loss.float()
    
            val_f1_score += f1_score(data[1].numpy(),val_pred,\
                                     average='macro', labels=np.unique(val_pred))
            val_loss += batch_loss.item()
            
        val_f1_score = val_f1_score/val_set.__len__() * 100
        train_f1_score = train_f1_score/train_set.__len__() * 100
        print('[%03d/%03d] %2.2f sec(s) Train F1_score: %3.3f Loss: %3.3f | Val F1_score: %3.3f loss: %3.3f' % \
                (epoch + 1, epochs, time.time()-epoch_start_time, \
                 train_f1_score, train_loss, val_f1_score, val_loss))
        
        loss_history.append((train_loss,val_loss))
        f1_score_history.append((train_f1_score,val_f1_score))
                
        if (val_f1_score > best_f1_score):
            torch.save(model.state_dict(), MODEL_PATH+'/'+str(iteration+1)+'_model.pth')
            best_f1_score = val_f1_score
            print ('Model Saved!')  
    return np.asarray(loss_history), np.asarray(f1_score_history)

def load_all_models(n_numbers, in_dim):
    all_models = list()
    for i in range(n_numbers):
        filename = MODEL_PATH+'/'+str(i+1)+'_model.pth'
        model = Classifier.My_Classifier(in_dim).cuda()
        model.load_state_dict(torch.load(filename))
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

def Evaluate(test_set,members):
    test_loader = DataLoader(test_set, batch_size=256, num_workers=WORKERS)
    stack_pred= []
    for model in members:
        pred = []
        for data in test_loader:
            test_pred = model(data[0].cuda())
            pred += [np.argmax(test_pred.cpu().data.numpy(), axis=1)]
        stack_pred.append(np.hstack(pred))
    
    # Voting
    stack_pred = np.asarray(stack_pred)
    y_pred = []
    for i in range(np.shape(stack_pred)[1]):
        y_pred += [np.argmax(np.bincount(stack_pred[:,i]))]
    test_f1_score = np.sum(np.array(y_pred) == test_set[:][1].numpy())
#    test_f1_score += f1_score(data[1].numpy(), batch_pred,\
#                                    average='macro', labels=np.unique(batch_pred))            
    test_f1_score = test_f1_score/test_set.__len__() * 100
    print("Test f1_score: %.3f" %test_f1_score,'%')


def _Ensemble():
    ## Read Raw Data
    raw = Data() 
    raw.Read(RAWDATA_PATH) 
    ## Encoding and Dim Reduction
    print('Encoding with discrete data...')
    enc_members = {}
    for arg in args_discrete:
        enc = OneHotEncoder(handle_unknown='ignore')
        raw.data[arg] = enc.fit_transform(raw.data[arg]).toarray() + E # sparse matrix to array
        enc_members[arg] = enc
    
    ## Unspervised learning - PCA(dimension reduction) with batch
    print('PCA to dimension reduction...')
    pca_members = {}
    from sklearn.decomposition import IncrementalPCA
    for arg in args_discrete:
        print(arg,'before pca', np.shape(raw.data[arg]))
        N_COMP = int(np.shape(raw.data[arg])[1] / 2)
        ipca = IncrementalPCA(n_components=N_COMP, copy=False, batch_size=BATCH_SIZE)
        raw.data[arg] = ipca.fit_transform(raw.data[arg])
        print(arg,'after dim reduce', np.shape(raw.data[arg]))
        pca_members[arg] = ipca
    
    ## Make the Torch
    X_train, y_train = [], []
    for arg in args_feautre:
        X_train += [raw.data[arg]]
        
    X_train, y_train = torch.FloatTensor(np.hstack(X_train)), torch.LongTensor(raw.target).view(-1,)
    
    ## Save pkl
    Save_PKL(X_train,X_train_PKL)
    Save_PKL(y_train,Y_train_PKL)
    Save_PKL(enc_members, './dataset/enc_members.pkl')
    Save_PKL(pca_members, './dataset/pca_members.pkl')
    
    ## Load pkl
    X_train = Load_PKL(X_train_PKL)
    y_train = Load_PKL(Y_train_PKL)
    enc_members = Load_PKL('./dataset/enc_members.pkl')
    pca_members = Load_PKL('./dataset/pca_members.pkl')
    
    X_train, X_test, y_train, y_test = Split_train_val(X_train, y_train, train_rate=0.8)
    
    for i in range(ENSEMBLE_NUM): 
        ## Split to training, validation  
        X_train_en, X_val, y_train_en, y_val = Split_train_val(X_train, y_train, train_rate=0.8)
        train_set, val_set = TensorDataset(X_train_en, y_train_en), TensorDataset(X_val, y_val)
       
        print("Building Model...")
        model = Classifier.My_Classifier(in_dim=X_train.size()[1]).cuda()
        loss = nn.BCELoss() # The criterion combines nn.LogSoftmax()
        loss = loss.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)   
        
        ## Training infomation
        print('Total parameters:',sum(p.numel() for p in model.parameters()),\
              ', Trainable parameters:comp',sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Training set size:',len(X_train),', Validation set size:',len(X_val))
        
        print("Fitting Model...")
        loss_history, acc_history = Fit(i,train_set,val_set,model,loss,optimizer,batch_size=BATCH_SIZE,epochs=EPOCHS)
    
    print('Training feature size', X_train.size()[1])
    members = load_all_models(ENSEMBLE_NUM, X_train.size()[1])
    
    print("Evaluate the model...") 
    test_set = TensorDataset(X_test, y_test)
    Evaluate(test_set,members)
    
#    print("======================= Deal with T-Brain test data... =======================")
#    ob = Data()
#    ob.Read(OBSERVE_PATH,'mean')
#    
#    ## Encoding and Dim Reduction
#    print('Encoding with discrete data...')
#    for arg in args_discrete:
#        enc = enc_members[arg]
#        ob.data[arg] = enc.transform(ob.data[arg]).toarray() + E # sparse matrix to array
#    
#    ## Unspervised learning - PCA(dimension reduction) with batch
#    print('PCA to dimension reduction...')
#    for arg in args_discrete:
#        print(arg,'before pca', np.shape(ob.data[arg]))
#        ipca = pca_members[arg]
#        ob.data[arg] = ipca.transform(ob.data[arg])
#        print(arg,'after dim reduce', np.shape(ob.data[arg]))
#    
#    ## Make the Torch
#    X_ob = []
#    for arg in args_feautre:
#        X_ob += [ob.data[arg]]
#        
#    X_ob = torch.FloatTensor(np.hstack(X_ob))
#    
#    ## Save set to pkl
#    Save_PKL(X_ob,X_ob_PKL)
#    ## Load pkl
##    X_ob = Load_PKL(X_ob_PKL)
#    
#    print('Submission...')
#    ob_set = TensorDataset(X_ob)
#    ob_loader = DataLoader(ob_set, batch_size=BATCH_SIZE, num_workers=WORKERS)
#    stack_pred= []
#    for model in members:
#        pred = []
#        for data in ob_loader:
#            ob_pred = model(data[0].cuda())
#            pred += [np.argmax(ob_pred.cpu().data.numpy(), axis=1)]
#        stack_pred.append(np.hstack(pred))
#    # Voting
#    stack_pred = np.asarray(stack_pred)
#    submission = []
#    for i in range(np.shape(stack_pred)[1]):
#        submission += [np.argmax(np.bincount(stack_pred[:,i]))]    
#    
#    with open(SUBMISSION_PATH, 'w') as submissionFile:
#        submissionFile.write('txkey,fraud_ind\n') 
#        for i,value in enumerate(submission):
#            submissionFile.write('%d,%d\n' %(ob.data['txkey'][i], value))
#    
#    print('======================= Writing Complete! =======================')    

if __name__ == '__main__':    
    _Ensemble()