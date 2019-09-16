import csv
import time
import pickle
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sample.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
PKL_PATH = './'
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
WORKERS = 0
TestCol = 10240
LEARNING_RATE = 0.001
######################### 

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
                
    def Split(self,md='split',rate=0.8):
        sp, sp_y = {}, []
        size = len(self.target)
        split_pos = int(np.round(size*rate))
        perm = np.random.permutation(size)  
        if md == 'split':
            for arg in self.data.keys():
                sp[arg] = self.data[arg][perm]             
                sp[arg] = sp[arg][split_pos:]            
                self.data[arg] = self.data[arg][:split_pos]       
            sp_y = self.target[perm]
            sp_y = sp_y[split_pos:]
            self.target = self.target[:split_pos]
#        if md == 'sample':
        return Data(sp, sp_y)
               
    def Make_Torch(self,*args,return_y=True):   
        self.X, self.y = [], []
        for arg in args:
            self.X += [self.data[arg]]
        self.X = torch.FloatTensor(np.hstack(self.X))
        if return_y:
            self.y = torch.LongTensor(self.target).view(-1,)
    
    def Pickle(self, name, md='save'):
        if md == 'save':
            file = open(name, 'wb')
            pickle.dump(self.data, file)
            file.close()

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
 
def Dim_Reduce(dataset,code_dim,epochs,*args):
    import FeatureExtract as fe
    for arg in args:
        print('encoding:',arg)
        encoded = torch.FloatTensor(dataset.data[arg])
        dataloader = DataLoader(encoded, batch_size=encoded.size()[0], shuffle=True) 
        model =  fe.AutoEncoder(encoded.size()[1], code_dim).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)    
        for epoch in range(1,epochs+1):
            for data in dataloader:
                output = model(data.cuda())
                loss = criterion(output, data.cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch,epochs,loss.item()))
        dataset.data[arg] = model.code.data.cpu().numpy() # dimension reduction
           
#def Dim_Reduce(x,code_dim,epochs,*args):
#    import FeatureExtract as fe
#    encoded = torch.FloatTensor(x)
##    dataloader = DataLoader(encoded, batch_size=1, shuffle=True) 
#    model =  fe.AutoEncoder(encoded.size()[1], code_dim).cuda()
#    criterion = nn.MSELoss()
#    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)    
#    for epoch in range(1,epochs+1):
#        output = model(encoded.cuda())
#        loss = criterion(output, encoded.cuda())
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#    print('epoch [{}/{}], loss:{:.4f}'.format(epoch,epochs,loss.item()))
#    dataset.data[arg] = model.code.data.cpu().numpy() # dimension reduction

def Split_train_val(X,y,split_rate=0.8):
    if len(X) != len(y):
        return print("no match size!")
    perm = torch.randperm(len(y)) # Shuffle
    X, y = X[perm], y[perm]
    split_pos = int(np.round(len(y)*split_rate))
    return X[:split_pos], X[split_pos:], y[:split_pos], y[split_pos:]

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs """
    lr = LEARNING_RATE * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Fit(train_set,val_set,model,loss,optimizer,batch_size,epochs):
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=WORKERS)   
    
    loss_history, acc_history = [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        adjust_learning_rate(optimizer, epoch)
        
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
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
            
            train_pred = train_pred.float()
            loss = loss.float()
            
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
     
        model.eval() # Switch to evaluate mode
        for i, data in enumerate(val_loader):
            # compute output
            with torch.no_grad():
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())
    
            val_pred = val_pred.float()
            batch_loss = batch_loss.float()
    
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            
        val_acc = val_acc/val_set.__len__()
        train_acc = train_acc/train_set.__len__()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.3f Loss: %3.3f | Val Acc: %3.3f loss: %3.3f' % \
                (epoch + 1, epochs, time.time()-epoch_start_time, \
                 train_acc, train_loss, val_acc, val_loss))
        
        loss_history.append((train_loss,val_loss))
        acc_history.append((train_acc,val_acc))
                
        if (val_acc > best_acc):
            torch.save(model.state_dict(), MODE_PATH+'/model.pth')
            best_acc = val_acc
            print ('Model Saved!')  
    return np.asarray(loss_history), np.asarray(acc_history)

def Parameters_Count(model):  
    print('Total parameters:',sum(p.numel() for p in model.parameters()))
    print('Trainable parameters:comp',sum(p.numel() for p in model.parameters() if p.requires_grad))
 
def Plot_History(history,save_model):
    plt.clf()
    loss_t, loss_v = history[:,0], history[:,1]
    plt.plot(loss_t,'b')
    plt.plot(loss_v,'r')
    if "loss" in save_model:
        plt.legend(['loss', 'val_loss'], loc="upper left")
        plt.ylabel("loss")
    elif "acc" in save_model:
        plt.legend(['acc', 'val_acc'], loc="upper left")
        plt.ylabel("acc")        
    plt.xlabel("epoch")
    plt.title("Training Process")
    if save_model == False:     
        plt.savefig(PIC_PATH+'/_history.png')
    else:
        plt.savefig(PIC_PATH+'/'+save_model+'_history.png')
    plt.show()
    plt.close()
    
def Evaluate(test_set,classifier):
    test_loader = DataLoader(test_set, num_workers=WORKERS)
    acc = 0.0
    y_pred = []
    classifier.eval()
    for i, data in enumerate(test_loader):
            test_pred = classifier(data[0].cuda())
            batch_pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            acc+=np.sum(batch_pred == data[1].numpy())
            y_pred.append(batch_pred)
    print("Test Accuracy: %.3f" %(acc/test_set.__len__()) )    


## Sample
# Read RAW data
args = 'bacno','cano','contp','etymd','mchno','acqic','mcc',\
     'ecfg','insfg','stocn','scity','stscd','ovrlt','flbmk','hcefg',\
     'csmcu','flg_3dsmk'
    
raw = Data() 
raw.Read(RAWDATA_PATH) 
raw.Encoding('OneHot',*args) # One-Hot encoding
#raw.Encoding('Label',*args)
test = raw.Split('split',0.8) # Split test data

# Normalize
mean, std = Normalize(raw.data,*args) 
Normalize(test.data,*args,mean_or_max=mean,std_or_min=std,trans=True)

# Feature Extract - Auto Encoder
Dim_Reduce(raw,8,32,'acqic')
Dim_Reduce(raw,16,32,'bacno','cano','mchno')

# Transform to tensor and save pkl
raw.Make_Torch(*FEATURE)
raw.Pickle(PKL_PATH+'train.pkl','save')

X_train, X_val, y_train, y_val = Split_train_val(raw.X, raw.y, 0.8)
train_set, val_set = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)

# Train Binary Classifier
import Classifier
model = Classifier.Binary_Classifier(X_train.size()[1]).cuda()
loss = nn.CrossEntropyLoss() # The criterion combines nn.LogSoftmax()
loss = loss.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)    

print("Fitting Model...")
Parameters_Count(model)
loss_history, acc_history = Fit(train_set,val_set,model,loss,optimizer,batch_size=4,epochs=2)

Plot_History(loss_history,'Classifier_loss')
Plot_History(acc_history,'Classifier_acc') 

# Evaluate model
# Feature Extract - Auto Encoder
Dim_Reduce(test, 8,32,'acqic')
Dim_Reduce(test, 16,32,'bacno','cano','mchno')

test.Make_Torch(*FEATURE)
test.Pickle(PKL_PATH+'test.pkl','save')

test_set = TensorDataset(test.X, test.y)
Evaluate(test_set,model)



## Submission
#ob = Data()
#ob.Read(OBSERVE_PATH)
#ob.Encoding('OneHot','acqic','csmcu')
#Pad(ob.data, mean,'acqic','csmcu','bacno')
#Normalize(ob.data,'acqic','csmcu','bacno',mean_or_max=mean,std_or_min=std,trans=True)
#ob.Make_Torch('acqic','csmcu','bacno',return_y=False)
#ob.Pickle(PKL_PATH+'test.pkl','save')

