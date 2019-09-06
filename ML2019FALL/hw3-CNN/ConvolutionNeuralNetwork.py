import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sample.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODE_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'
CLASSES = ("Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral")
##################

def Imshow(img):
    import matplotlib.pyplot as plt
    npimg = img.numpy()[0] # Show one picture
    plt.imshow(npimg)

class Data():
    def __init__(self):
        self.X = []
        self.y = []
    def Read(self,path):
        raw = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
        for i in range(len(raw)):
                tmp = np.array(raw[i, 1].split(' ')).reshape(1, 48, 48) # (RGB-channel, height, width)
                self.X.append(tmp)
                self.y.append(np.array(raw[i, 0]))
        self.X = torch.FloatTensor(np.array(self.X, dtype=float))
        self.y = torch.LongTensor(np.array(self.y, dtype=int)) 

def Split_train_val(x,y,train_rate=0.7):
    if len(x) != len(y):
        return print("no match size!")
    perm = torch.randperm(len(y)) # Shuffle
    x, y = x[perm], y[perm]
    split_pos = int(np.round(len(y)*train_rate))
    return x[:split_pos], x[split_pos:], y[:split_pos], y[split_pos:]
    
def Normalization(x,method='ZoomOut',mean_or_max='None',std_or_min='None',trans=False):
    if trans == False:
        if method == 'Rescaling':
            mean_or_max = torch.max(x, dim = 0) # (Number of Images, RGB, height, width)
            std_or_min = torch.min(x, dim = 0)  
        elif method == 'Standardization':
            mean_or_max = torch.mean(x, dim = 0)
            std_or_min = torch.std(x, dim = 0)          
    if method == 'Rescaling':
        x = (x - std_or_min) / (mean_or_max - std_or_min)
    elif method == 'Standardization':
        x = (x-mean_or_max) / std_or_min
    elif method == 'ZoomOut':
        x /= 255.0
    if trans:
        return x
    else:
        return x, mean_or_max, std_or_min

def Encoding(y,n_class,method='One-Hot'):
    size = len(y)
    if method == 'One-Hot':
        encoding = torch.LongTensor(size,n_class)
        encoding.zero_()
        encoding.scatter_(1,y.view(-1,1),1)
    return encoding

def ImageDataGenerator(imgs, label):
    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]
    aug, aug_label = imgs.clone(), label.clone()
    for i in range(len(aug)):
        aug[i] = flip(aug[i],2) # horizontal_flip
    imgs, label = torch.cat((imgs,aug)), torch.cat((label,aug_label))
    return imgs, label
    
'''
    Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
    dilation=1, groups=1, bias=True, padding_mode='zeros') 
        in_channels(int) - Number of channels in the input image (1:binary , 3: RGB)
        out_channels(int) - Number of channels produced by the convolution,
                            i.e., the number of feature filter(neuron) will be conneted.
        
    BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    LeakyReLU(negative_slope=0.01, inplace=False)
    
    MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    Dropout(p=0.5, inplace=False)
'''

class VGG16(nn.Module):
    def __init__(self, in_dim, n_class):
        super(VGG16, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),           
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),            
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),        
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),       
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), # BatchNorm before activation
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),       
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)             
        )

        self.fc = nn.Sequential(
            nn.Linear(512*1*1, 4096), # need to check Linear input size
            nn.ReLU(True),
            nn.Dropout(), 
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(), 
            nn.Linear(1000, n_class)
        )
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) # Flatten
        return self.fc(out)




  
def Parameters_Count(model,select):  
    if select == 'total':
        print(sum(p.numel() for p in model.parameters()))
    elif select == 'trainable':
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

  
def Fit(train_set,val_set,model,loss,optimizer,batch_size,epochs):
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)   
    
    loss_history, acc_history = [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
    
        model.train() # Switch to train mode
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
    
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()
    
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
     
        model.eval() # Switch to evaluate mode
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())
    
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        
        val_acc = val_acc/val_set.__len__()
        train_acc = train_acc/train_set.__len__()
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, epochs, time.time()-epoch_start_time, \
                 train_acc, train_loss, val_acc, val_loss))
        
        loss_history.append((train_loss,val_loss))
        acc_history.append((train_acc,val_acc))
                
        if (val_acc > best_acc):
            torch.save(model.state_dict(), MODE_PATH+'/model.pth')
            best_acc = val_acc
            print ('Model Saved!')  
    return np.asarray(loss_history), np.asarray(acc_history)

def Evaluate(test_set,classifier):
    test_loader = DataLoader(test_set, num_workers=8)
    acc = 0.0
    y_pred = []
    classifier.eval()
    for i, data in enumerate(test_loader):
            test_pred = classifier(data[0].cuda())
            batch_pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            acc+=np.sum(batch_pred == data[1].numpy())
            y_pred.append(batch_pred)
    print("Test Accuracy: %.3f" %(acc/test_set.__len__()) )
    Plot_Confusion_Matrix(list(test_set[:][1].numpy()), y_pred)
    
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

def Plot_Confusion_Matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import itertools
    conf_matrix = confusion_matrix(y_true, y_pred)
    title='Normalized Confusion Matrix'
    cm = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i,j]), horizontalalignment="center", 
                 color="white" if cm[i,j] > thresh else "black")    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return 

#def Cross_validation(raw, n_fold):
#    X_train, X_val, y_train, y_val = Split_train_val(raw.X, raw.y, train_rate=0.7)
#    train_set = TensorDataset(raw.X, raw.y)
    
#def Ensemble_Learning():


def Sample():
    print("Reading Raw File...")
    raw = Data()
    raw.Read(RAWDATA_PATH)
    raw.X, raw.y = ImageDataGenerator(raw.X, raw.y) # return tensor
    raw.X,_,_ = Normalization(raw.X,'ZoomOut') 
    
    ## Split to training, validation and testing set
    X_train, X_test, y_train, y_test = Split_train_val(raw.X, raw.y, train_rate=0.9)
    X_train, X_val, y_train, y_val = Split_train_val(X_train, y_train, train_rate=0.9)
    
    ## Make the training and testing dataset
    train_set, val_set = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    
    print("Building Model...")
    model = VGG16(in_dim=1, n_class=7).cuda()
    loss = nn.CrossEntropyLoss() # The criterion combines nn.LogSoftmax()
    loss = loss.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)    
    
    print("Fitting Model...")
    loss_history, acc_history = Fit(train_set,val_set,model,loss,optimizer,batch_size=256,epochs=32)
    
    Plot_History(loss_history,'VGG16_loss')
    Plot_History(acc_history,'VGG16_acc')  
    
    ## It does't work, beacuse that would make the acc to random, fix it!
#    print("Loading Model...")
#    model = VGG16(in_dim=1, n_class=7).cuda()
#    model.load_state_dict(torch.load(MODE_PATH+'/model.pth'))
    
    print("Evaluate and Plot Confusion Matrix...")
    Evaluate(test_set,model)
    
    print("Reading Observe File...")
    ob = Data()
    ob.Read(OBSERVE_PATH)
    ob.X,_,_ = Normalization(ob.X,'ZoomOut')
    
    ob_set = TensorDataset(ob.X, ob.y) ## test.y is ID not label.
    ob_loader = DataLoader(ob_set, num_workers=8)
    
    print("Submission...")
    submission = [['id', 'label']]
    for i, data in enumerate(ob_loader):
            test_pred = model(data[0].cuda())
            pred = np.argmax(test_pred.cpu().data.numpy(), axis=1)[0]
            submission.append([i, pred])
    with open(SUBMISSION_PATH, 'w') as submissionFile:
        writer = csv.writer(submissionFile)
        writer.writerows(submission)
    
    print('Writing Complete!')

if __name__ == '__main__':    
    Sample()