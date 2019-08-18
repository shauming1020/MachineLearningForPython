# HW1-Linear Regression
import sys
import numpy as np
import pandas as pd
import csv

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sampleSubmission.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODE_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'

FEATURES = {'AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5',
            'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR'}
##################

def Create_dict(raw, features):
    n_features = len(features)
    raw = raw[1:,3:]
    where_are_NaNs = np.isnan(raw)
    raw[where_are_NaNs] = 0
    month_to_data = {}
    for month in range(12):
        sample = np.empty(shape = (18 , 480))
        for day in range(20):
            for hour in range(24): 
                sample[:,day * 24 + hour] = raw[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
        month_to_data[month] = sample
    return month_to_data

def Preprocess(month_to_data):
    ## Preprocess
    # Sample every 10 hrs: previous 9-hr as train-feature, 10th-hr as train-predict
    # Every month's 0~9 hrs cannot be predict.
    x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
    y = np.empty(shape = (12 * 471 , 1),dtype = float)
    
    for month in range(12): 
        for day in range(20): 
            for hour in range(24):   
                if day == 19 and hour > 14:
                    continue  
                x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1) 
                y[month * 471 + day * 24 + hour,0] = month_to_data[month][9 ,day * 24 + hour + 9]
    return x,y

def Normalization(x,y,mean=0,std=0):
    ## Normalization
    mean = np.mean(x, axis = 0) 
    std = np.std(x, axis = 0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    return x,y,mean,std

def Split_train_test(x,y,train_rate=0.7):
#    perm = np.random.permutation(len(y))
#    x, y = x[perm], y[perm]
    split_pos = np.round(len(y)*train_rate).astype('int32')
    return x[:split_pos], x[split_pos:], y[:split_pos], y[split_pos:]

def Fit(X,y,val=None,validation_split=0.0,bias=True,weights='zeros',
      loss='rmse',opt='vanilla',regularization=None,lamda=0.0005,learning_rate=200,
      epochs=1,batch_size=None,patience=None,save_model=True):

    def initial(X,weights,bias):    
        dim = X.shape[1]
        if bias:
            dim = dim + 1
            X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1).astype(float)
        if weights == 'zeros':
            w = np.zeros(shape=(dim, 1))
        elif weights == 'random':
            w = np.random.rand(dim, 1)
        return X, w
    # loss_value      
    def loss_value(X,y,w,loss):
        if loss == 'mse':
            pred = X.dot(w)
            square_error = (y-pred)**2
            mse = square_error.sum()/len(y)
            return mse
        elif loss == 'rmse':
            mse = loss_value(X,y,w,'mse')
            return np.power(mse,0.5)
    # Optmize function
    def update_parameters(X,y,w,sigma,opt):
        if regularization == 'l1':
            reg = lamda * w
        elif regularization == 'l2':
            reg = 2. * lamda * w
        elif regularization == None:
            reg = lamda
        pred = X.dot(w)
        grad = -2. * np.dot(X.T,y-pred) + reg
        if opt == 'ada':
            sigma += np.sqrt(grad**2)
            w -= grad * learning_rate / sigma 
        else:
            w -= grad * learning_rate
        return w, sigma
    
    def get_mini_batch(X,y,batch_size):
        mini_batches = []
        data = np.hstack((X,y))
        np.random.shuffle(data) 
        n_minibatches = data.shape[0] 
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        return mini_batches

    # Split Training and Validation Set
    if val:
        X_t,X_v,y_t,y_v = Split_train_test(X,y,(1-validation_split))
    else:
        X_t,X_v,y_t,y_v = X,X,y,y
    # Initialization
    X_t, w = initial(X_t,weights,bias)
    X_v, w_v = initial(X_v,weights,bias)  
    # Set model parameter
    sigma = None
    if opt == 'stg':
        batch_size = 1
    elif opt == 'ada':
        sigma = np.zeros(w.shape)   
    # Mini-Batch Training    
    cost_history = []
    see_cost,see = None,0
    best_w = w
    for epoch in range(epochs):
        # Get batch data
        mini_batches = get_mini_batch(X_t, y_t, batch_size)
        for mini_batch in mini_batches: 
            X_mini, y_mini = mini_batch 
            # Update weights
            w, sigma = update_parameters(X_mini,y_mini,w,sigma,opt)
            # Cost
            cost = loss_value(X_t,y_t,w,loss) # on training set
            cost_v = loss_value(X_v,y_v,w,loss) # on validation ser
            cost_history.append((cost,cost_v))       
        print('Epoch:'+str(epoch) + '/' + str(epochs),'-loss: %.3f' % cost," val loss: %.3f" % cost_v)
        # Early Stopping
        if patience != None:
            see+=1
            if see_cost == None:
                see_cost = cost
            elif see_cost > cost:
                continue
            else:
                print('from ',cost,' imporve to ',see_cost)
                see_cost = cost
                see = 0
                best_w = w
            if see >= patience:
                print('early stopping',see,'/',patience)
                break
        # Save best model
        if save_model:
            np.save('weight.npy',best_w) # save weight
    return w, cost_history


def Stacked_Ensemble(NUM):
    ## Read in raw data
    with open(RAWDATA_PATH) as f:
        raw = np.genfromtxt(f,delimiter=',')
    ## Create the data dictory
    month_to_data = Create_dict(raw)
    ## Preprocess to split features and true data
    X, y = Preprocess(month_to_data)
    ## Normalization
    X,y,mean,std = Normalization(X,y)
    ## Split to training and testing set
    X_train, X_test, y_train, y_test = Split_train_test(X,y,train_rate=0.9)
    
    ##  Training
    val=True
    validation_split=0.1
    bias=True
    weights='zeros'
    loss='rmse'
    opt='ada'
    regularization='l2'
    lamda= 0.01
    learning_rate=200
    epochs=10000
    batch_size=1024
    patience=None
    save_model=True
    
    model, history = Fit(X_train,y_train,val,validation_split,bias,weights,
                      loss,opt,regularization,lamda,learning_rate,
                      epochs,batch_size,patience,save_model)
    
    ## Read in Observe set
    model = np.load('weight.npy')                                ## load weight
    test_raw_data = np.genfromtxt(OBSERVE_PATH, delimiter=',')   ## test.csv
    test_data = test_raw_data[:, 2: ]
    where_are_NaNs = np.isnan(test_data)
    test_data[where_are_NaNs] = 0 
    
    test_x = np.empty(shape = (240, 18 * 9),dtype = float)
    for i in range(240):
        test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 

    test_x = np.empty(shape = (240, 18 * 9),dtype = float)
    
    for i in range(240):
        test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 
    
    
    for i in range(test_x.shape[0]):        ##Normalization
        for j in range(test_x.shape[1]):
            if not std[j] == 0 :
                test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]
    
    test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
    answer = test_x.dot(model)  

    ## Write file
    with open(SUBMISSION_PATH,"w") as f:
        w = csv.writer(f)
        title = ['id','value']
        w.writerow(title) 
        for i in range(240):
            content = ['id_'+str(i),answer[i][0]]
            w.writerow(content) 

if __name__ == '__main__':    
    Stacked_Ensemble(1)